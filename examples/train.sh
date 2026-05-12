python train_itd_hdemucs_musdb.py \
  --data_root /path/to/binaural_musdb18hq \
  --subset train \
  --batch_size 1 \
  --epochs 7 \
  --lr 1e-4 \
  --lambda_itd 0.1 \
  --segment_seconds 6.0 \
  --device cuda
  