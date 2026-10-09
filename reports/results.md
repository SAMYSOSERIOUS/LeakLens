# Audit results


Data: 2015-01-01 to 2017-03-31, 2,426,143 customers, 22,975,416 transactions.  
Test month (cut-off): **2017-01-31** · threshold tuned on 2016-11-30 · trained on 2016-07-31, 2016-08-31, 2016-09-30, 2016-10-31.

## 1–3. Public notebooks, re-tested

| Public notebook | What it claimed | Re-run as published (AUC) | Honest re-test (AUC) | Drop | Main leak |
|---|---|---|---|---|---|
| [jsroa15/KKBOX (random forest)](https://github.com/jsroa15/KKBOX) | Random forest, ROC AUC 0.940 on its test set (README). | 0.974 | 0.779 | −0.195 | `mode_quarter`, `regist_cancels`, `revenue` |
| [apostaremczak/churn-prediction (random forest)](https://github.com/apostaremczak/churn-prediction) | 'Unbalanced' random forest: accuracy 92.7%, F1 0.48 (model_results.json). | 0.989 | 0.826 | −0.163 | `membership_expire_date`, `transaction_date`, `is_cancel` |
| [naomifridman/Deep-VAE-prediction-of-churn-customer (VAE + KNN)](https://github.com/naomifridman/Deep-VAE-prediction-of-churn-customer) | KNN on a VAE latent space: accuracy 95.0%, churn-class F1 0.66 (notebook output). | 0.973 | 0.804 | −0.169 | `membership_expire_date`, `transaction_date`, `is_cancel` |

### A. jsroa15/KKBOX (random forest)

Claimed: Random forest, ROC AUC 0.940 on its test set (README).

| Setting | AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| As published (whole file, random split) | 0.974 | 0.972 | 0.610 | 0.663 | 0.564 |
| Leak-free features, still random split | 0.839 | 0.948 | 0.263 | 0.298 | 0.235 |
| Honest (cut-off date, train on past, test on future) | 0.779 | 0.951 | 0.238 | 0.304 | 0.195 |

Which features changed when the future was removed:

| Feature | Customers whose value changed | Signal with future data | Signal without |
|---|---|---|---|
| `mode_quarter` | 51.3% | 0.664 | 0.565 |
| `regist_cancels` | 5.9% | 0.583 | 0.518 |
| `revenue` | 97.4% | 0.676 | 0.630 |
| `regist_trans` | 97.4% | 0.767 | 0.722 |
| `tenure` | 85.1% | 0.549 | 0.531 |
| `is_auto_renew` | 7.9% | 0.780 | 0.778 |
| `mode_plan_days` | 0.6% | 0.530 | 0.529 |
| `avg_num_50` | 78.1% | 0.545 | 0.545 |
| `mode_payment_method` | 1.1% | 0.740 | 0.740 |
| `avg_num_75` | 78.0% | 0.542 | 0.542 |
| `avg_num_25` | 78.3% | 0.534 | 0.535 |
| `avg_num_985` | 78.0% | 0.535 | 0.537 |
| `avg_num_unq` | 78.3% | 0.526 | 0.531 |
| `avg_total_secs` | 78.2% | 0.516 | 0.522 |
| `avg_num_100` | 78.2% | 0.512 | 0.518 |

- Cancellation count, auto-renew share, revenue and tenure are computed over the whole transactions file.
- Churners are oversampled in training, as in the original.

### B. apostaremczak/churn-prediction (random forest)

Claimed: 'Unbalanced' random forest: accuracy 92.7%, F1 0.48 (model_results.json).

| Setting | AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| As published (whole file, random split) | 0.989 | 0.993 | 0.914 | 0.943 | 0.886 |
| Leak-free features, still random split | 0.830 | 0.958 | 0.184 | 0.379 | 0.121 |
| Honest (cut-off date, train on past, test on future) | 0.826 | 0.956 | 0.172 | 0.330 | 0.117 |

Which features changed when the future was removed:

| Feature | Customers whose value changed | Signal with future data | Signal without |
|---|---|---|---|
| `membership_expire_date` | 20.0% | 0.952 | 0.519 |
| `transaction_date` | 8.1% | 0.950 | 0.562 |
| `is_cancel` | 3.5% | 0.654 | 0.526 |
| `actual_amount_paid` | 5.0% | 0.723 | 0.714 |
| `plan_list_price` | 4.7% | 0.726 | 0.725 |
| `is_auto_renew` | 1.1% | 0.771 | 0.771 |
| `payment_plan_days` | 2.1% | 0.562 | 0.563 |
| `payment_method_id` | 2.6% | 0.738 | 0.742 |

- Uses each user's last row of transactions_v2, which runs to the end of the month in which churn is decided.

### C. naomifridman/Deep-VAE-prediction-of-churn-customer (VAE + KNN)

Claimed: KNN on a VAE latent space: accuracy 95.0%, churn-class F1 0.66 (notebook output).

| Setting | AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| As published (whole file, random split) | 0.973 | 0.973 | 0.551 | 0.790 | 0.423 |
| Leak-free features, still random split | 0.840 | 0.961 | 0.149 | 0.542 | 0.086 |
| Honest (cut-off date, train on past, test on future) | 0.804 | 0.961 | 0.101 | 0.546 | 0.056 |

Which features changed when the future was removed:

| Feature | Customers whose value changed | Signal with future data | Signal without |
|---|---|---|---|
| `membership_expire_date` | 20.0% | 0.952 | 0.519 |
| `transaction_date` | 8.1% | 0.950 | 0.562 |
| `is_cancel` | 3.5% | 0.654 | 0.526 |
| `trans_count` | 97.4% | 0.767 | 0.722 |
| `logs_count` | 78.4% | 0.558 | 0.537 |
| `total_secs` | 78.3% | 0.536 | 0.518 |
| `num_100` | 77.6% | 0.536 | 0.518 |
| `num_unq` | 78.4% | 0.536 | 0.519 |
| `num_25` | 77.2% | 0.529 | 0.512 |
| `num_75` | 73.7% | 0.530 | 0.514 |
| `num_985` | 73.7% | 0.530 | 0.514 |
| `num_50` | 75.0% | 0.525 | 0.510 |
| `actual_amount_paid` | 5.0% | 0.723 | 0.714 |
| `plan_list_price` | 4.7% | 0.726 | 0.725 |
| `is_auto_renew` | 1.1% | 0.771 | 0.771 |
| `payment_plan_days` | 2.1% | 0.562 | 0.563 |
| `payment_method_id` | 2.6% | 0.738 | 0.742 |

- Substitution: we run the KNN (k=27) on scaled features directly instead of on a VAE's 2-D compression. The leakage sits in the inputs, which we keep exactly.

## 4. Honest models on the test month

| Model | AUC | PR-AUC | Log loss | Saved at money threshold | Saved at 0.5 |
|---|---|---|---|---|---|
| Nobody churns (baseline) | 0.500 | 0.040 | 0.170 | €0 (t=0.06) | €0 |
| Auto-renew rule | 0.799 | 0.186 | 0.131 | €909,470 (t=0.02) | €97,310 |
| Logistic regression | 0.866 | 0.404 | 0.116 | €989,575 (t=0.11) | €289,225 |
| LightGBM | 0.872 | 0.444 | 0.112 | €1,024,505 (t=0.09) | €297,125 |

## 5–6. Money

Assumptions: offer costs €5, a lost customer costs €60, offer keeps 100% of the leavers it reaches. Break-even chance of leaving: 8.3%.

| Who gets an offer | Threshold | Offers | Leavers reached | Leavers missed | Saved |
|---|---|---|---|---|---|
| money-based threshold | 0.09 | 81,371 | 23,856 | 11,022 | €1,024,505 |
| default 0.5 | 0.50 | 7,619 | 5,587 | 29,291 | €297,125 |
| offer to everyone | 0.00 | 883,727 | 34,878 | 0 | −€2,325,955 |
| no offers | 1.00 | 0 | 0 | 34,878 | €0 |

Best possible threshold in hindsight: 0.08 (€1,027,745). Ours was picked a month earlier, without seeing the answers.

Most useful inputs for LightGBM (share of total gain):

- `payment_method`: 35.0%
- `auto_renew`: 23.2%
- `avg_paid`: 9.8%
- `days_active_m1`: 5.1%
- `days_active_m3`: 4.9%
- `cancel_last`: 4.2%
- `unique_songs_m1`: 3.9%
- `amount_paid`: 2.5%
- `n_tx_180d`: 2.5%
- `list_price`: 2.4%
