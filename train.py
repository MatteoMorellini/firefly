"""LoRA fine-tune CLIP on (image, Q&A) pairs with symmetric InfoNCE.

Per training step we sample one Q&A per image, so each image appears at most
once in a batch (avoiding in-batch self-collision among negatives).

Targets: nn.Linear modules inside the visual and text transformer blocks
(out_proj of attention, c_fc / c_proj of MLP). LoRA params + logit_scale are
trained; everything else is frozen.
"""

import argparse
import json
import math
import random
import re
from datetime import datetime
from pathlib import Path

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

QA_RE = re.compile(r"Q:\s*(.*?)\s*A:\s*(.*)", re.DOTALL)


def parse_qa(entry: str):
    m = QA_RE.match(entry.strip())
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Wrap an nn.Linear with a low-rank update. Original weights are frozen."""

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
        # B starts at zero so the wrapped layer initially equals the base.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return out + self.scaling * update

    # open_clip introspects .weight / .bias on inner Linears (e.g. for dtype).
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


# Module-name suffixes inside open_clip transformer blocks worth adapting.
LORA_TARGETS = ("out_proj", "c_fc", "c_proj")


def inject_lora(model: nn.Module, r: int, alpha: float, dropout: float):
    n_replaced = 0
    # Collect first so we don't mutate during iteration.
    targets = []
    for name, module in model.named_modules():
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


class QADataset(Dataset):
    """One item per image per epoch — one Q&A descriptor sampled at random."""

    def __init__(self, items, image_dir: Path, preprocess, rng: random.Random):
        self.items = [
            (name, [parse_qa(e) for e in entries if parse_qa(e)])
            for name, entries in items
        ]
        self.items = [(n, qas) for n, qas in self.items if qas]
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.rng = rng

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name, qas = self.items[idx]
        question, answer = self.rng.choice(qas)
        image = self.preprocess(Image.open(self.image_dir / name).convert("RGB"))
        return image, f"Q: {question} A: {answer}"


def collate(batch, tokenizer):
    images = torch.stack([b[0] for b in batch])
    texts = tokenizer([b[1] for b in batch])
    return images, texts


# ---------------------------------------------------------------------------
# Evaluation helpers (inlined to avoid circular import with main.py)
# ---------------------------------------------------------------------------

YES_NO = ["yes", "no"]
NUMBERS = [str(i) for i in range(0, 11)]
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
SIZES = ["small", "large", "very small"]
MATERIALS = [
    "brick",
    "checkered",
    "chessboard",
    "circles",
    "emojis",
    "metal",
    "rubber",
    "star",
    "wave",
    "zigzag",
]


def classify_question(q: str) -> str:
    ql = q.lower().strip().rstrip("?.!")
    if re.search(r"\bwhat (?:color|colour)\b", ql) or "what color is" in ql:
        return "color"
    if re.search(r"\bwhat (?:is the )?shape\b", ql) or "what shape is" in ql:
        return "shape"
    if "made of" in ql or re.search(r"\bwhat (?:is the )?material\b", ql):
        return "material"
    if re.search(r"\bwhat size\b", ql):
        return "size"
    if ql.startswith(("how many", "what number of")):
        return "number"
    if ql.startswith(
        (
            "is there",
            "are there",
            "is the",
            "are any",
            "is any",
            "are the",
            "does",
            "do",
        )
    ):
        return "yes_no"
    return "other"


def candidates_for(category: str, gold: str):
    gold_l = gold.lower().strip()
    base = {
        "yes_no": list(YES_NO),
        "number": list(NUMBERS),
        "color": list(COLORS),
        "shape": list(SHAPES),
        "material": list(MATERIALS),
        "size": list(SIZES),
    }.get(category, YES_NO + NUMBERS + COLORS + SHAPES + MATERIALS + SIZES)
    if gold_l not in base:
        base.append(gold_l)
    return base


class EvalDataset(Dataset):
    """One item per (image, Q&A) pair for batched evaluation."""

    def __init__(self, items, image_dir: Path, preprocess):
        self.pairs = []
        for img_name, entries in items:
            img_path = image_dir / img_name
            if not img_path.exists():
                continue
            for entry in entries:
                parsed = parse_qa(entry)
                if parsed:
                    question, gold = parsed
                    self.pairs.append((img_path, question, gold))
        self.preprocess = preprocess

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, question, gold = self.pairs[idx]
        image = self.preprocess(Image.open(img_path).convert("RGB"))
        return image, question, gold


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    preprocess,
    device,
    eval_items,
    image_dir: Path,
    batch_size: int = 64,
):
    model.eval()
    total = correct = 0
    per_type = {
        k: [0, 0]
        for k in ["yes_no", "number", "color", "shape", "material", "size", "other"]
    }

    dataset = EvalDataset(eval_items, image_dir, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    for images, questions, golds in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        img_feats = model.encode_image(images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        for img_feat, question, gold in zip(img_feats, questions, golds):
            category = classify_question(question)
            cands = candidates_for(category, gold)

            tokens = tokenizer([f"Q: {question} A: {c}" for c in cands]).to(device)
            txt_feat = model.encode_text(tokens)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            pred = cands[int((img_feat @ txt_feat.T).squeeze(0).argmax().item())]
            is_correct = pred.lower() == gold.lower()
            per_type[category][0] += int(is_correct)
            per_type[category][1] += 1
            total += 1
            correct += int(is_correct)

    accuracy = correct / total if total else 0.0
    per_type_acc = {
        k: v[0] / v[1] if v[1] else 0.0 for k, v in per_type.items() if v[1] > 0
    }
    return accuracy, per_type_acc


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
    parser.add_argument("--images", type=Path, default=Path("./images"))
    parser.add_argument("--questions", type=Path, default=Path("./questions.json"))
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Cap on images.")
    parser.add_argument(
        "--amp", action="store_true", help="Use mixed-precision on CUDA."
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("./runs"),
                        help="Parent directory for run folders.")
    parser.add_argument(
        "--eval",
        type=Path,
        default=None,
        help="Path to eval split JSON. If set, evaluate periodically.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Run evaluation every N epochs (default: 5).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.model.replace('/', '-')}"
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}")

    print(f"Loading {args.model} ({args.pretrained}) on {args.device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(args.device)

    # Freeze everything, then inject LoRA, then re-enable logit_scale.
    for p in model.parameters():
        p.requires_grad_(False)
    n_lora = inject_lora(
        model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout
    )
    model = model.to(args.device)  # move newly-created LoRA params
    model.logit_scale.requires_grad_(True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"injected LoRA into {n_lora} linear modules")
    print(
        f"trainable params: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)"
    )

    with open(args.questions) as f:
        data = json.load(f)
    items = list(data.items())
    if args.limit:
        items = items[: args.limit]

    eval_items = None
    if args.eval:
        with open(args.eval) as f:
            eval_items = list(json.load(f).items())

    rng = random.Random(args.seed)
    dataset = QADataset(items, args.images, preprocess, rng)
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
            },
        }

    model.train()
    best_loss = float("inf")
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

            running += loss.item()
            if step % 5 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = running / max(1, len(loader))
        msg = f"epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}  logit_scale={model.logit_scale.exp().item():.2f}"

        if eval_items and (epoch + 1) % args.eval_every == 0:
            acc, per_type_acc = evaluate(
                model, tokenizer, preprocess, args.device, eval_items, args.images
            )
            per_type_str = "  ".join(f"{k}={v:.2%}" for k, v in per_type_acc.items())
            msg += f"  eval_acc={acc:.2%}  [{per_type_str}]"
            model.train()

        if avg < best_loss:
            best_loss = avg
            torch.save(make_state(), run_dir / "best.pt")
            msg += "  [best]"

        print(msg)

    torch.save(make_state(), run_dir / "last.pt")
    n_lora_params = sum(v.numel() for v in lora_state_dict(model).values())
    print(f"best loss={best_loss:.4f}  run dir: {run_dir}  ({n_lora_params:,} LoRA params)")


if __name__ == "__main__":
    main()
