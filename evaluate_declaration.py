"""Evaluate CLIP (optionally LoRA-tuned) on the declaration variant of CLEVR.

For each scene in statements.json we pick the model's best guess for every
feature (color, shape, material, count) by ranking a canonical sentence per
candidate value via CLIP similarity, then compare to the scene's gold values.
"""

import argparse
import json
from pathlib import Path

import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from train_feature import (
    FEATURES,
    feature_values,
    inject_lora,
    render,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("./images"))
    parser.add_argument("--statements", type=Path, default=Path("./statements.json"))
    parser.add_argument("--image-suffix", default=".png")
    parser.add_argument("--split", default=None,
                        help="If set, only evaluate scenes with this split value.")
    parser.add_argument("--model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="laion2b_s32b_b82k")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("./results.json"))
    parser.add_argument("--lora", type=Path, default=None,
                        help="Path to a lora.pt saved by train_feature.py / train.py.")
    args = parser.parse_args()

    print(f"Loading {args.model} ({args.pretrained}) on {args.device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(args.device)

    if args.lora:
        ckpt = torch.load(args.lora, map_location=args.device)
        cfg = ckpt.get("config", {})
        inject_lora(
            model,
            r=cfg.get("lora_r", 8),
            alpha=cfg.get("lora_alpha", 16.0),
            dropout=0.0,
        )
        model = model.to(args.device)
        missing, unexpected = model.load_state_dict(ckpt["lora"], strict=False)
        loaded = sum(1 for k in ckpt["lora"] if k not in unexpected)
        print(f"loaded {loaded} LoRA tensors from {args.lora} "
              f"(unexpected={len(unexpected)})")
        if "logit_scale" in ckpt:
            with torch.no_grad():
                model.logit_scale.copy_(ckpt["logit_scale"].to(args.device))

    model.eval()

    with open(args.statements) as f:
        data = json.load(f)

    items = list(data.items())
    if args.split:
        items = [(n, p) for n, p in items if p.get("split") == args.split]
    if args.limit:
        items = items[: args.limit]

    # Pre-encode candidate texts that don't depend on the scene's count.
    # color/shape phrasing depends on singular/plural, so we cache by count.
    feature_text_cache: dict[tuple[str, int], torch.Tensor] = {}

    def encode_candidates(feature: str, count: int) -> torch.Tensor:
        key = (feature, count if feature in ("color", "shape") else 0)
        if key in feature_text_cache:
            return feature_text_cache[key]
        values = feature_values(feature)
        prompts = [render(feature, v, count) for v in values]
        tokens = tokenizer(prompts).to(args.device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feature_text_cache[key] = feats
        return feats

    total = 0
    correct = 0
    per_feature = {f: [0, 0] for f in FEATURES}
    results = []

    with torch.no_grad():
        for img_name, payload in tqdm(items, desc="eval"):
            img_path = args.images / f"{img_name}{args.image_suffix}"
            if not img_path.exists():
                print(f"missing: {img_path}")
                continue
            image = (
                preprocess(Image.open(img_path).convert("RGB"))
                .unsqueeze(0)
                .to(args.device)
            )
            img_feat = model.encode_image(image)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            values = payload["values"]
            count = int(values["count"])
            scene_result = {"image": img_name, "split": payload.get("split"), "predictions": {}}

            for feature in FEATURES:
                cands = feature_values(feature)
                txt_feat = encode_candidates(feature, count)

                sims = (img_feat @ txt_feat.T).squeeze(0)
                best = int(sims.argmax().item())
                pred = cands[best]

                gold = values[feature]
                gold_norm = int(gold) if feature == "count" else str(gold)
                is_correct = pred == gold_norm

                per_feature[feature][0] += int(is_correct)
                per_feature[feature][1] += 1
                total += 1
                correct += int(is_correct)

                scene_result["predictions"][feature] = {
                    "gold": gold_norm,
                    "pred": pred,
                    "correct": is_correct,
                }

            results.append(scene_result)

    summary = {
        "model": args.model,
        "pretrained": args.pretrained,
        "lora": str(args.lora) if args.lora else None,
        "split": args.split,
        "scenes": len(results),
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "per_feature": {
            f: {
                "correct": v[0],
                "total": v[1],
                "accuracy": v[0] / v[1] if v[1] else 0.0,
            }
            for f, v in per_feature.items()
        },
    }
    print(json.dumps(summary, indent=2))

    args.output.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2)
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
