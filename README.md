source venv/bin/activate
pip install -r requirements.txt

python train_vs_xgboost.py \
  --train_csv ../data/processed/train.csv \
  --val_csv ../data/processed/val.csv \
  --test_csv ../data/processed/test.csv \
  --train_emb ../data/embeddings/train_emb.npy \
  --val_emb ../data/embeddings/val_emb.npy \
  --test_emb ../data/embeddings/test_emb.npy \
  --dialect_col dialect_strict \
  --out_dir ../data/results/vs_xgb_train_time

  python train_adv_xgb.py