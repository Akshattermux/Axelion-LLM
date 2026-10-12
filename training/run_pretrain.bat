@echo off
echo === Axelion Pretraining Launcher ===
if not exist logs mkdir logs
python -m training.pretrain ^
  --size 1b ^
  --train-bin data/pretraining/train.bin ^
  --val-bin data/pretraining/validation.bin ^
  --checkpoint-out checkpoints/axelion-base-1b.pt ^
  --max-iters 10000 ^
  --batch-size 1 ^
  --grad-accum 16 ^
  --lr 3e-4 ^
  --min-lr 3e-5 ^
  --warmup-steps 500 ^
  --eval-interval 250 ^
  --seed 42 ^
  --log-file logs/pretrain.log
pause
