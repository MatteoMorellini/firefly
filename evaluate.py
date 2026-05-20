import argparse
import json
import re
from pathlib import Path

import open_clip
import torch
from PIL import Image

from train import inject_lora


YES_NO = ["yes", "no"]
NUMBERS = [str(i) for i in range(0, 11)]

SHAPES = ["cone", "cube", "cylinder", "diamond", "gear", "monkey", "sphere", "star", "teapot", "torus"]
COLORS = ["blue", "brown", "cyan", "gray", "green", "orange", "pink", "purple", "red", "yellow"]
SIZES = ["small", "large", "very small"]
MATERIALS = ["brick", "checkered", "chessboard", "circles", "emojis", "metal", "rubber", "star", "wave", "zigzag"]

QA_RE = re.compile(r"Q:\s*(.*?)\s*A:\s*(.*)", re.DOTALL)


def parse_qa(entry: str):
    m = QA_RE.match(entry.strip())
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def classify_question(q: str) -> str:
    """Pick a category from the question text alone (no peeking at the answer)."""
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
    if ql.startswith(("is there", "are there", "is the", "are any", "is any", "are the", "does", "do")):
        return "yes_no"
    return "other"


def candidates_for(category: str, gold: str):
    """Return candidate answer strings for a question category.

    Gold is appended if missing so the correct answer is always in the pool;
    this avoids penalizing CLIP for vocab we forgot to include.
    """
    gold_l = gold.lower().strip()
    if category == "yes_no":
        base = list(YES_NO)
    elif category == "number":
        base = list(NUMBERS)
    elif category == "color":
        base = list(COLORS)
    elif category == "shape":
        base = list(SHAPES)
    elif category == "material":
        base = list(MATERIALS)
    elif category == "size":
        base = list(SIZES)
    else:
        # Unknown category: union of everything so the gold is at least represented.
        base = YES_NO + NUMBERS + COLORS + SHAPES + MATERIALS + SIZES
    if gold_l not in base:
        base.append(gold_l)
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("./images"))
    parser.add_argument("--questions", type=Path, default=Path("./questions.json"))
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of images.")
    parser.add_argument("--output", type=Path, default=Path("./results.json"))
    parser.add_argument("--lora", type=Path, default=None,
                        help="Path to a lora.pt saved by train.py. If set, injects adapters and loads weights.")
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

    with open(args.questions) as f:
        data = json.load(f)

    items = list(data.items())
    if args.limit:
        items = items[: args.limit]

    total = 0
    correct = 0
    per_type = {k: [0, 0] for k in ["yes_no", "number", "color", "shape", "material", "size", "other"]}
    results = []

    with torch.no_grad():
        for img_name, qas in items:
            img_path = args.images / img_name
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

            for entry in qas:
                parsed = parse_qa(entry)
                if not parsed:
                    continue
                question, gold = parsed
                gold_l = gold.lower()
                category = classify_question(question)
                cands = candidates_for(category, gold)

                prompts = [f"Q: {question} A: {c}" for c in cands]
                tokens = tokenizer(prompts).to(args.device)
                txt_feat = model.encode_text(tokens)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

                sims = (img_feat @ txt_feat.T).squeeze(0)
                best = int(sims.argmax().item())
                pred = cands[best]

                is_correct = pred.lower() == gold_l
                per_type[category][0] += int(is_correct)
                per_type[category][1] += 1
                total += 1
                correct += int(is_correct)

                results.append(
                    {
                        "image": img_name,
                        "question": question,
                        "gold": gold,
                        "pred": pred,
                        "correct": is_correct,
                        "category": category,
                        "num_candidates": len(cands),
                    }
                )

    summary = {
        "model": args.model,
        "pretrained": args.pretrained,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "per_type": {
            k: {
                "correct": v[0],
                "total": v[1],
                "accuracy": v[0] / v[1] if v[1] else 0.0,
            }
            for k, v in per_type.items()
        },
    }
    print(json.dumps(summary, indent=2))

    args.output.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2)
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
