# Data

Nothing in this folder is committed except this note.

## Real data: KKBox (WSDM 2018 churn challenge)

1. Download the files from Kaggle: https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data
   (free account; accept the competition rules first). Note: the Mendeley Data page
   (https://data.mendeley.com/datasets/mv3f8bdrvy) only has `transactions_v2.csv`, which
   mostly covers March 2017 and is not enough for an honest month-by-month test.
   Never open these files in Excel: saving cuts them off at 1,048,575 rows.
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
