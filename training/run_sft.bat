@echo off
echo === Axelion SFT Launcher ===
if not exist logs mkdir logs
python -m training.sft ^
  --base-ckpt checkpoints/axelion-base-mini.pt ^
  --output-ckpt checkpoints/axelion-instruct-v1.pt ^
  --train-jsonl data/instruction/train.jsonl ^
  --val-jsonl data/instruction/validation.jsonl ^
  --epochs 3 ^
  --lr 3e-5 ^
  --batch-size 4 ^
  --grad-accum 4 ^
  --log-file logs/sft.log
pause
