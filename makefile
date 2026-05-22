TODO: #1 reformat to use variables for features
eval-color:
	uv run python evaluate_declaration.py \
		--images ./clevr_4/images \
		--statements ./data/declarations/test.json \
		--output results_ws_color.json \
		--lora runs/20260520_150644_ViT-B-32_color/best.pt

train-material: 
	uv run python train_feature.py \
	--feature material \
	--epochs 100 \
	--images clevr_4/images/ \
	--statements ./data/declarations/train.json \
	--eval-statements data/declarations/eval.json

train-shape: 
	uv run python train_feature.py \
	--feature shape \
	--epochs 100 \
	--images clevr_4/images/ \
	--statements ./data/declarations/train.json \
	--eval-statements data/declarations/eval.json

train-color: 
	uv run python train_feature.py \
	--feature color \
	--epochs 100 \
	--images clevr_4/images/ \
	--statements ./data/declarations/train.json \
	--eval-statements data/declarations/eval.json

train-count: 
	uv run python train_feature.py \
	--feature count \
	--epochs 100 \
	--images clevr_4/images/ \
	--statements ./data/declarations/train.json \
	--eval-statements data/declarations/eval.json