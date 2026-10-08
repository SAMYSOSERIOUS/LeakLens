# Data

Nothing in this folder is committed except this note.

## Real data: KKBox (WSDM 2018 churn challenge)

1. Download the files from Mendeley Data: https://data.mendeley.com/datasets/mv3f8bdrvy
   (a mirror of the Kaggle "WSDM - KKBox's Churn Prediction Challenge" files).
2. Unpack these into `data/kkbox/raw/`:
   - `transactions.csv`, `transactions_v2.csv`
   - `members_v3.csv`
   - `user_logs.csv`, `user_logs_v2.csv` (about 30 GB unpacked; optional but recommended)
3. Run `make prepare`. It reads the raw files once (the logs in chunks, so it
   works on a laptop) and writes compact tables to `data/kkbox/prepared/`.
4. In `config.toml` set `prepared_dir = "data/kkbox/prepared"` and `is_sample = false`.

The official `train.csv` / `train_v2.csv` label files are not needed. We rebuild
the churn label for every month ourselves with the same rule (no new paid
subscription within 30 days after the membership runs out), because an honest
time split needs labels for many months, not one.

## Demo data

`make sample` writes a small made-up dataset in exactly the same file format to
`data/sample/`. Tests, CI and the weekly demo job use it. Its numbers are not
KKBox results.
