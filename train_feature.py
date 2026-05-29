"""LoRA fine-tune CLIP on (image, statement) pairs restricted to a single
feature (color, shape, material, or count) of the modified CLEVR variant
where every object in a scene shares the same shape, material and color.

Training:
- statements.json (from generate_statements.py) supplies multiple phrasings
  per scene per category. Each epoch samples one phrasing for the chosen
  feature per image, then runs symmetric InfoNCE.

Evaluation:
- The same statements.json embeds the raw scene values used as gold labels.
  For each image, build a candidate sentence per possible feature value
  using one of the training templates; pick the argmax.
"""

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from clevr_4.declaration_utils.generate_statements import (
    COLOR_TEMPLATES_PLURAL,
    COLOR_TEMPLATES_SINGULAR,
    COUNT_TEMPLATES_PLURAL,
    COUNT_TEMPLATES_SINGULAR,
    MATERIAL_TEMPLATES,
    SHAPE_TEMPLATES_PLURAL,
    SHAPE_TEMPLATES_SINGULAR,
    pluralize,
)
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

FEATURES = ("color", "shape", "material", "count")


# ---------------------------------------------------------------------------
# LoRA (copied verbatim from train.py)
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return out + self.scaling * update

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


LORA_TARGETS = ("out_proj", "c_fc", "c_proj")


def inject_lora(model: nn.Module, r: int, alpha: float, dropout: float):
    n_replaced = 0
    targets = []
    for _, module in model.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            if not child_name.endswith(LORA_TARGETS):
                continue
            targets.append((module, child_name, child))
    for parent, child_name, child in targets:
        setattr(
            parent, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
        )
        n_replaced += 1
    return n_replaced


def lora_state_dict(model: nn.Module):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class FeatureStatementDataset(Dataset):
    """One item per image per epoch — one phrasing sampled from the feature's pool."""

    def __init__(self, items, feature: str, image_dir: Path, preprocess, rng):
        self.items = [
            (name, payload["statements"][feature])
            for name, payload in items
            if payload["statements"].get(feature)
        ]
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.rng = rng
        self.image_suffix = ".png"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name, phrasings = self.items[idx]
        text = self.rng.choice(phrasings)
        image_path = self.image_dir / f"{name}{self.image_suffix}"
        image = self.preprocess(Image.open(image_path).convert("RGB"))
        return image, text


def collate(batch, tokenizer):
    images = torch.stack([b[0] for b in batch])
    texts = tokenizer([b[1] for b in batch])
    return images, texts


# ---------------------------------------------------------------------------
# Feature vocabularies + canonical phrasing for eval
# ---------------------------------------------------------------------------


COLORS = [
    "blue",
    "brown",
    "cyan",
    "gray",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "yellow",
]
SHAPES = [
    "cone",
    "cube",
    "cylinder",
    "diamond",
    "gear",
    "monkey",
    "sphere",
    "star",
    "teapot",
    "torus",
]
MATERIALS = [
    "brick",
    "checkered",
    "chessboard",
    "circles",
    "emojis",
    "metal",
    "rubber",
    "wave",
    "zigzag",
]
COUNTS = list(range(1, 11))


def feature_values(feature: str):
    return {
        "color": COLORS,
        "shape": SHAPES,
        "material": MATERIALS,
        "count": COUNTS,
    }[feature]


def render(feature: str, value, count: int) -> str:
    """Render one canonical phrasing for (feature, value) given the scene's count.

    Uses the first template from the appropriate pool so eval is deterministic.
    """
    if feature == "material":
        return MATERIAL_TEMPLATES[0].format(texture=value)
    if feature == "count":
        pool = COUNT_TEMPLATES_SINGULAR if value == 1 else COUNT_TEMPLATES_PLURAL
        return pool[0].format(count=value)
    if feature == "shape":
        if count == 1:
            return SHAPE_TEMPLATES_SINGULAR[0].format(shape=value)
        return SHAPE_TEMPLATES_PLURAL[0].format(
            shape=value, shape_plural=pluralize(value)
        )
    if feature == "color":
        pool = COLOR_TEMPLATES_SINGULAR if count == 1 else COLOR_TEMPLATES_PLURAL
        return pool[0].format(color=value)
    raise ValueError(feature)


class FeatureEvalDataset(Dataset):
    def __init__(self, items, feature, image_dir, preprocess, image_suffix):
        self.feature = feature
        self.items = []
        for name, payload in items:
            img_path = image_dir / f"{name}{image_suffix}"
            if not img_path.exists():
                continue
            self.items.append((img_path, payload["values"]))
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, values = self.items[idx]
        image = self.preprocess(Image.open(img_path).convert("RGB"))
        return image, values[self.feature], values["count"]


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    preprocess,
    device,
    items,
    image_dir,
    feature,
    image_suffix,
    batch_size=64,
):
    model.eval()
    dataset = FeatureEvalDataset(items, feature, image_dir, preprocess, image_suffix)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    values = feature_values(feature)
    total = correct = 0

    for images, golds, counts in tqdm(loader, desc=f"eval[{feature}]", leave=False):
        images = images.to(device)
        img_feats = model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        for img_feat, gold, count in zip(img_feats, golds, counts):
            count = int(count)
            cands = [render(feature, v, count) for v in values]
            tokens = tokenizer(cands).to(device)
            txt_feat = model.encode_text(tokens)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            pred_idx = int((img_feat @ txt_feat.T).squeeze(0).argmax().item())
            pred = values[pred_idx]
            gold_norm = int(gold) if feature == "count" else str(gold)
            correct += int(pred == gold_norm)
            total += 1

    return correct / total if total else 0.0


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def contrastive_loss(image_features, text_features, logit_scale):
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = logit_scale * image_features @ text_features.T
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature",
        choices=FEATURES,
        required=True,
        help="Which scene feature to train on.",
    )
    parser.add_argument("--images", type=Path, default=Path("./images"))
    parser.add_argument(
        "--statements",
        type=Path,
        default=Path("./train.json"),
        help="Training split (output of split.py / generate_statements.py)",
    )
    parser.add_argument(
        "--eval-statements",
        type=Path,
        default=None,
        help="Eval split (e.g. eval.json). Defaults to --statements if not set.",
    )
    parser.add_argument(
        "--image-suffix",
        default=".png",
        help="File extension to append to the image id.",
    )
    parser.add_argument("--model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b82k")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=Path("./runs"))
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--eval-every-steps",
        type=int,
        default=0,
        help="If > 0, also run evaluation every N training steps within an epoch.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_id = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_{args.model.replace('/', '-')}_{args.feature}"
    )
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}")
    print(f"feature: {args.feature}")

    print(f"Loading {args.model} ({args.pretrained}) on {args.device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(args.device)

    for p in model.parameters():
        p.requires_grad_(False)
    n_lora = inject_lora(
        model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    model = model.to(args.device)
    model.logit_scale.requires_grad_(True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"injected LoRA into {n_lora} linear modules")
    print(
        f"trainable params: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)"
    )

    with open(args.statements) as f:
        data = json.load(f)
    items = list(data.items())
    if args.limit:
        items = items[: args.limit]

    eval_path = args.eval_statements or args.statements
    with open(eval_path) as f:
        eval_data = json.load(f)
    eval_items = list(eval_data.items())

    rng = random.Random(args.seed)
    dataset = FeatureStatementDataset(items, args.feature, args.images, preprocess, rng)
    dataset.image_suffix = args.image_suffix
    print(f"train scenes: {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate(b, tokenizer),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    use_amp = args.amp and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def make_state():
        return {
            "lora": lora_state_dict(model),
            "logit_scale": model.logit_scale.detach().cpu(),
            "config": {
                "model": args.model,
                "pretrained": args.pretrained,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "feature": args.feature,
            },
        }

    train_log = open(run_dir / "train_log.jsonl", "a")
    eval_log = open(run_dir / "eval_log.jsonl", "a")

    def run_eval(epoch: int, step: int, global_step: int):
        acc = evaluate(
            model,
            tokenizer,
            preprocess,
            args.device,
            eval_items,
            args.images,
            args.feature,
            args.image_suffix,
        )
        eval_log.write(
            json.dumps(
                {
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "feature": args.feature,
                    "accuracy": acc,
                }
            )
            + "\n"
        )
        eval_log.flush()
        model.train()
        return acc

    model.train()
    best_loss = float("inf")
    global_step = 0
    for epoch in range(args.epochs):
        running = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", miniters=5)
        for step, (images, texts) in enumerate(pbar, 1):
            images = images.to(args.device, non_blocking=True)
            texts = texts.to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                image_features = model.encode_image(images)
                text_features = model.encode_text(texts)
                logit_scale = model.logit_scale.exp().clamp(max=100.0)
                loss = contrastive_loss(image_features, text_features, logit_scale)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            loss_val = loss.item()
            running += loss_val
            train_log.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "step": step,
                        "global_step": global_step,
                        "loss": loss_val,
                        "lr": scheduler.get_last_lr()[0],
                        "logit_scale": model.logit_scale.exp().item(),
                    }
                )
                + "\n"
            )
            train_log.flush()
            if step % 5 == 0:
                pbar.set_postfix(loss=f"{loss_val:.4f}")

            if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                acc = run_eval(epoch + 1, step, global_step)
                pbar.write(
                    f"  step {global_step}: eval_acc[{args.feature}]={acc:.2%}"
                )

        avg = running / max(1, len(loader))
        msg = (
            f"epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}  "
            f"logit_scale={model.logit_scale.exp().item():.2f}"
        )

        acc = run_eval(epoch + 1, len(loader), global_step)
        msg += f"  eval_acc[{args.feature}]={acc:.2%}"

        torch.save(make_state(), run_dir / f"{epoch + 1}.pt")

        if avg < best_loss:
            best_loss = avg
            torch.save(make_state(), run_dir / "best.pt")
            msg += "  [best]"

        print(msg)

    train_log.close()
    eval_log.close()

    torch.save(make_state(), run_dir / "last.pt")
    n_lora_params = sum(v.numel() for v in lora_state_dict(model).values())
    print(
        f"best loss={best_loss:.4f}  run dir: {run_dir}  ({n_lora_params:,} LoRA params)"
    )


if __name__ == "__main__":
    main()
