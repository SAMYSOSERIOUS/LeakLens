# Three popular churn models, re-tested honestly: from 0.98 to 0.74 AUC

[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)
[![weekly replay](../../actions/workflows/weekly-replay.yml/badge.svg)](../../actions/workflows/weekly-replay.yml)

Companies spend real money on retention offers ("stay and get a month free") based on churn
models. Those models often look much better in testing than they work in real life, because
the test lets them peek at the future. **LeakLens** measures that gap on three popular public
models, builds an honest one, and turns its scores into a decision in euros: who is actually
worth an offer.

**The question:** how much do popular public churn models overstate their performance, and
which customers is it actually worth sending a retention offer to?

## Results

<!-- RESULTS:START -->
| Public notebook | What it claimed | Re-run as published (AUC) | Honest re-test (AUC) | Drop | Main leak |
|---|---|---|---|---|---|
| [jsroa15/KKBOX (random forest)](https://github.com/jsroa15/KKBOX) | Random forest, ROC AUC 0.940 on its test set (README). | 0.977 | 0.743 | −0.233 | `mode_quarter`, `regist_cancels`, `revenue` |
| [apostaremczak/churn-prediction (random forest)](https://github.com/apostaremczak/churn-prediction) | 'Unbalanced' random forest: accuracy 92.7%, F1 0.48 (model_results.json). | 0.989 | 0.826 | −0.163 | `membership_expire_date`, `transaction_date`, `is_cancel` |
| [naomifridman/Deep-VAE-prediction-of-churn-customer (VAE + KNN)](https://github.com/naomifridman/Deep-VAE-prediction-of-churn-customer) | KNN on a VAE latent space: accuracy 95.0%, churn-class F1 0.66 (notebook output). | 0.977 | 0.808 | −0.169 | `membership_expire_date`, `transaction_date`, `is_cancel` |

Our honest LightGBM on the same test month: AUC **0.865** (auto-renew rule 0.799, nobody-churns baseline 0.500). Money-based threshold **0.09** saves **€1,007,035** on 883,727 customers, versus €126,795 at the default 0.5. Full tables: [reports/results.md](reports/results.md).
<!-- RESULTS:END -->

**What these numbers cover.** Full KKBox transaction and member data (2.4 million customers,
23 million transactions, Jan 2015 – Mar 2017). Test month: customers whose membership ran out
in February 2017, scored on 31 January 2017 (883,727 customers, 3.9% churned). The listening
logs (30 GB) were left out of this run, so the features use payments and profiles only.
The three public notebooks are re-tested on a random sample of 300,000 of those customers to
keep their slow models practical; our own models use all of them. The churn rate is lower
than in the competition because we score every customer whose membership ends that month,
including the many auto-renewers.

- One-page summary for managers: [docs/manager_summary.md](docs/manager_summary.md)
- All tables, per-feature leak checks: [reports/results.md](reports/results.md)
- Live monitor page: published by GitHub Pages from [`site/`](site/) (enable Pages → "GitHub Actions")

## Why this dataset

There is no good public German churn dataset: German companies don't publish customer data,
and data protection rules (GDPR) make that unlikely to change. So this project uses the
**KKBox** data from the WSDM 2018 churn challenge
([Kaggle](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data)). KKBox is a Taiwanese
music streaming service. The data fits this project for three reasons:

- **Real subscription transactions with dates** for millions of users (payments, plan
  length, auto-renew, cancellations, membership end dates) plus daily listening logs.
- **Dates make an honest split possible.** You can pick a cut-off date, build features only
  from before it, and measure churn only after it, month after month.
- **It is the dataset the popular public notebooks used**, so their published scores can be
  re-tested on the same data.

Churn follows the official challenge rule: a customer has churned if no new paid
subscription starts within 30 days after their membership runs out.

## What "honest" means here

```
         features from here only            churn measured here only
 ───────────────────────────────────────┤├──────────────────────────────────▶ time
                                    cut-off
                                     date
```

A model used on Monday morning can only know what happened up to Sunday night. The audit
re-runs each public notebook three ways, which separates the two kinds of cheating:

1. **As published:** features from the whole data file, random 70/30 split of customers.
2. **Leak-free features, still random split:** same model, but every feature only uses data
   from before the cut-off. The drop from 1 to 2 is the cost of the **leaked features**.
3. **Honest:** leak-free features, trained on earlier months, tested on a later month it has
   never seen. Training months are only those whose answers were already known on the test
   date. The drop from 2 to 3 is the cost of **testing on the past instead of the future**.

To find *which* features leaked, each one is rebuilt with and without future data. The report
lists what share of customers got a different value and how much predictive power the
feature loses once the future is removed.

## The three public notebooks

| | Notebook | What it claims | What we kept / changed |
|---|---|---|---|
| A | [jsroa15/KKBOX](https://github.com/jsroa15/KKBOX) | Random forest, ROC AUC 0.940 on test | Same transaction, log and profile aggregates, oversampling, random forest |
| B | [apostaremczak/churn-prediction](https://github.com/apostaremczak/churn-prediction) | Random forest, accuracy 92.7%, F1 0.48 | Same last-transaction and member columns (dates as numbers), "unbalanced" random forest |
| C | [naomifridman/Deep-VAE-prediction-of-churn-customer](https://github.com/naomifridman/Deep-VAE-prediction-of-churn-customer) | VAE + KNN, accuracy 95.0%, F1 0.66 | Same inputs. **Substitution:** KNN (k=27) on scaled features instead of on a VAE's 2-D compression. The leak is in the inputs, which are unchanged. |

Claimed numbers come from each repository's README, results file or notebook output. The
recipes in [`src/leaklens/recipes.py`](src/leaklens/recipes.py) rebuild each notebook's
feature list, split and model from its published code and description, so that all three run
through the same harness on the same data. They are re-implementations, not byte-for-byte
copies, and the differences are listed above. These three were chosen because each one
publishes its code, its feature list and its claimed scores on GitHub. Re-testing popular
Kaggle competition notebooks the same way is a natural next step.

## Our honest model

Trained on four months, threshold tuned on the next known month, tested on a later month:

- **Nobody churns:** predicts the average churn rate for everyone (baseline, AUC 0.5).
- **Auto-renew rule:** what a business does without a model: customers who switched off
  auto-renew or just cancelled are "at risk".
- **Logistic regression** and **LightGBM** on 31 features: the latest known transaction,
  payment history (lapses, cancels, discounts), profile, and listening activity in the last
  one and three full months.

## Turning scores into euros

Assumptions (in [`config.toml`](config.toml), change them freely):

| | Value | Note |
|---|---|---|
| Cost of one offer | **€5** | paid for every customer who gets it |
| Value of a customer | **€60** | lost when a customer leaves |
| Share of leavers the offer keeps | **100%** | the brief's assumption; real offers keep far fewer, try 0.3 |

```
money saved = leavers reached × share kept × €60  −  offers sent × €5
```

An offer pays off when *chance of leaving × €60 > €5*, so the break-even chance is
**8.3%**, not the usual default of 50%. The audit picks the money-maximising cut-off on one
month and counts the money on the next, then compares it with the 0.5 default, offering to
everyone, and offering to nobody.

## Production layer: a simulated live business

The KKBox data doesn't grow, so [`src/leaklens/replay.py`](src/leaklens/replay.py) replays
history **one month per run**, as if each month had just arrived.

- **Scheduled job (weekly):** [`.github/workflows/weekly-replay.yml`](.github/workflows/weekly-replay.yml)
  runs every Monday. It moves the clock one month forward, scores every customer whose
  membership ends next month, checks predictions whose answers are now known (about two
  months later, because customers get 30 days to renew), and commits its state to `state/`.
  At the end of the data it starts the replay over. The real data is too big for the repo,
  so the job needs the prepared data as a download link in the secret `PREPARED_DATA_URL`;
  without it the job skips itself and the page keeps the last snapshot from a local run
  (`make replay-loop`).
- **Retrain rule:** if the current model's latest checked AUC is below
  `retrain_below_auc` (default 0.80), the job retrains on the newest months whose answers
  are known and logs why.
- **Drift warning:** each month the job compares this month's customers with the training
  customers, input by input (PSI score). Above 0.2 the page shows a warning. Inputs that grow
  for everyone every month (time as a customer) are left out, because they would raise a
  false alarm every run.
- **Web page:** [`site/index.html`](site/index.html), a static page (no server) with the
  audit verdict, accuracy month by month with retrain markers, drift, the cost calculator
  and the offer list.
- **Cost calculator:** two sliders (offer cost, customer value). The audit precomputes how
  many leavers and stayers fall in each 1%-wide band of predicted risk, so the page can
  recompute the best cut-off and the money saved for any setting without a server.

The demo data contains a planted change: from September 2016 a partner promotion brings in
customers on a free first month, and many leave when they first have to pay. In the replay
the drift warning fires in September, the drop in accuracy is confirmed two months later,
the job retrains, and accuracy recovers.

## Tests

`pytest` runs 30 checks (also on every push, via GitHub Actions):

- **Leakage (the most important):** for four cut-off dates, every future row is scrambled
  (fake cancels and renewals for every customer, changed payments, wild listening numbers)
  and **no feature value may change**. The same scramble **must** change the public
  notebooks' features, which proves the test can catch a leak. A third check runs the weekly
  job on real and scrambled futures and requires identical scores.
- **Model:** LightGBM beats "nobody churns" and the auto-renew rule on a later month (AUC
  and log loss).
- **Data:** every customer appears once per scoring date; labels are only 0 or 1; scored
  customers' memberships end in the following month; the churn rule gives the hand-worked
  answer on a four-customer example.
- **Money:** the savings formula gives the hand-checked answer on a tiny example
  (e.g. offers to 2 customers, 1 leaver: 1 × €60 − 2 × €5 = €50), and the page's banded
  calculation matches the exact one.

## Run it

```bash
pip install -e ".[dev]"      # Python 3.11+
make sample                   # made-up data in KKBox format (≈1 min)
make test                     # 30 tests
make audit                    # steps 1-6, writes reports/, docs/, README results
make replay-loop              # 15 months of the live simulation
make serve                    # monitor at http://localhost:8000
```

### Run it on the real data

Download `transactions.csv.7z`, `transactions_v2.csv.7z` and `members_v3.csv.7z` from the
[Kaggle competition page](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data)
(free account, accept the rules), unpack them into `data/kkbox/raw`, then:

```bash
make prepare
# in config.toml: prepared_dir = "data/kkbox/prepared", is_sample = false
make audit && make replay-loop
```

The audit takes about 30 minutes on a laptop. The public notebooks are re-tested on a random
sample of `max_train_rows` (300,000) customers to keep their slow models practical. To run the weekly job on real data in
GitHub Actions, upload the prepared folder as a `.tar.gz` and set its link as the secret
`PREPARED_DATA_URL`.

## Project layout

```
src/leaklens/
  dates.py      cut-off dates, when answers become known
  data.py       load raw KKBox files, compact them (logs summed per month)
  labels.py     who is scored on a date, and did they churn
  features.py   honest features; visible() is the only door into the data
  recipes.py    the three public notebooks, rebuilt
  models.py     nobody-churns, auto-renew rule, logistic regression, LightGBM
  money.py      savings, best threshold, banded counts for the page
  audit.py      steps 1-6
  replay.py     the weekly job: score, check, retrain, drift, page data
  report.py     README results, manager summary, full tables
  sample.py     made-up data in KKBox format (for tests and the demo)
tests/          leakage, model, data, money
site/           the static monitor page (+ data/ written by the job)
state/          the replay job's memory between runs
```

## Limits

- The public notebooks are rebuilt, not executed from their original files; see the table
  above for what was kept and changed.
- The €5 / €60 / 100% figures are assumptions from the brief. The offer's real effect
  should be measured with a random control group before trusting any euro figure.
- The real-data run leaves out the listening logs. Listening activity is one of the
  strongest honest signals, so our model's score is probably a lower bound. Adding
  `user_logs.csv` (30 GB) and rerunning `make prepare && make audit` includes it.
- One test month (February 2017 expiries). Repeating the audit on other months would show
  how stable the gaps are.
- KKBox is a Taiwanese music service from 2015–2017. The method transfers; the exact numbers
  won't.

MIT licence.
