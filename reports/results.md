# Audit results

> **Demo numbers.** These come from the built-in made-up sample data, not from KKBox. Run the pipeline on the real files (see *Run it on the real data*) to get real numbers.


Data: 2014-12-29 to 2017-03-31, 8,000 customers, 66,803 transactions.  
Test month (cut-off): **2017-01-31** · threshold tuned on 2016-11-30 · trained on 2016-07-31, 2016-08-31, 2016-09-30, 2016-10-31.

## 1–3. Public notebooks, re-tested

| Public notebook | What it claimed | Re-run as published (AUC) | Honest re-test (AUC) | Drop | Main leak |
|---|---|---|---|---|---|
| [jsroa15/KKBOX (random forest)](https://github.com/jsroa15/KKBOX) | Random forest, ROC AUC 0.940 on its test set (README). | 0.969 | 0.832 | −0.137 | `mode_quarter`, `regist_trans`, `tenure` |
| [apostaremczak/churn-prediction (random forest)](https://github.com/apostaremczak/churn-prediction) | 'Unbalanced' random forest: accuracy 92.7%, F1 0.48 (model_results.json). | 0.999 | 0.668 | −0.331 | `transaction_date`, `membership_expire_date`, `actual_amount_paid` |
| [naomifridman/Deep-VAE-prediction-of-churn-customer (VAE + KNN)](https://github.com/naomifridman/Deep-VAE-prediction-of-churn-customer) | KNN on a VAE latent space: accuracy 95.0%, churn-class F1 0.66 (notebook output). | 0.882 | 0.734 | −0.148 | `transaction_date`, `membership_expire_date`, `trans_count` |

### A. jsroa15/KKBOX (random forest)

Claimed: Random forest, ROC AUC 0.940 on its test set (README).

| Setting | AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| As published (whole file, random split) | 0.969 | 0.948 | 0.760 | 0.839 | 0.695 |
| Leak-free features, still random split | 0.818 | 0.891 | 0.429 | 0.571 | 0.343 |
| Honest (cut-off date, train on past, test on future) | 0.832 | 0.882 | 0.515 | 0.504 | 0.527 |

Which features changed when the future was removed:

| Feature | Customers whose value changed | Signal with future data | Signal without |
|---|---|---|---|
| `mode_quarter` | 37.0% | 0.795 | 0.636 |
| `regist_trans` | 90.2% | 0.757 | 0.623 |
| `tenure` | 90.2% | 0.739 | 0.615 |
| `is_auto_renew` | 10.9% | 0.629 | 0.570 |
| `regist_cancels` | 4.5% | 0.595 | 0.536 |
| `revenue` | 75.2% | 0.746 | 0.692 |
| `avg_num_50` | 97.5% | 0.531 | 0.502 |
| `avg_num_unq` | 97.6% | 0.712 | 0.702 |
| `avg_num_100` | 97.8% | 0.712 | 0.703 |
| `avg_total_secs` | 97.7% | 0.710 | 0.702 |
| `avg_num_25` | 97.5% | 0.553 | 0.545 |
| `avg_num_75` | 97.3% | 0.552 | 0.545 |
| `mode_payment_method` | 0.7% | 0.681 | 0.680 |
| `avg_num_985` | 0.5% | 0.500 | 0.500 |

- Cancellation count, auto-renew share, revenue and tenure are computed over the whole transactions file.
- Churners are oversampled in training, as in the original.

### B. apostaremczak/churn-prediction (random forest)

Claimed: 'Unbalanced' random forest: accuracy 92.7%, F1 0.48 (model_results.json).

| Setting | AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| As published (whole file, random split) | 0.999 | 0.998 | 0.990 | 1.000 | 0.981 |
| Leak-free features, still random split | 0.828 | 0.897 | 0.449 | 0.617 | 0.352 |
| Honest (cut-off date, train on past, test on future) | 0.668 | 0.884 | 0.212 | 0.541 | 0.132 |

Which features changed when the future was removed:

| Feature | Customers whose value changed | Signal with future data | Signal without |
|---|---|---|---|
| `transaction_date` | 18.4% | 0.980 | 0.512 |
| `membership_expire_date` | 2.7% | 0.985 | 0.530 |
| `actual_amount_paid` | 12.0% | 0.771 | 0.673 |
| `is_auto_renew` | 4.6% | 0.656 | 0.592 |
| `is_cancel` | 4.5% | 0.600 | 0.537 |

- Uses each user's last row of transactions_v2, which runs to the end of the month in which churn is decided.

### C. naomifridman/Deep-VAE-prediction-of-churn-customer (VAE + KNN)

Claimed: KNN on a VAE latent space: accuracy 95.0%, churn-class F1 0.66 (notebook output).

| Setting | AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| As published (whole file, random split) | 0.882 | 0.878 | 0.182 | 0.444 | 0.114 |
| Leak-free features, still random split | 0.764 | 0.892 | 0.188 | 0.917 | 0.105 |
| Honest (cut-off date, train on past, test on future) | 0.734 | 0.890 | 0.134 | 1.000 | 0.072 |

Which features changed when the future was removed:

| Feature | Customers whose value changed | Signal with future data | Signal without |
|---|---|---|---|
| `transaction_date` | 18.4% | 0.980 | 0.512 |
| `membership_expire_date` | 2.7% | 0.985 | 0.530 |
| `trans_count` | 90.2% | 0.757 | 0.623 |
| `actual_amount_paid` | 12.0% | 0.771 | 0.673 |
| `logs_count` | 97.9% | 0.749 | 0.683 |
| `num_50` | 97.5% | 0.748 | 0.683 |
| `is_auto_renew` | 4.6% | 0.656 | 0.592 |
| `num_75` | 97.8% | 0.749 | 0.685 |
| `num_25` | 97.9% | 0.748 | 0.684 |
| `is_cancel` | 4.5% | 0.600 | 0.537 |
| `num_unq` | 97.9% | 0.753 | 0.698 |
| `total_secs` | 97.9% | 0.753 | 0.698 |
| `num_100` | 97.8% | 0.753 | 0.703 |

- Substitution: we run the KNN (k=27) on scaled features directly instead of on a VAE's 2-D compression. The leakage sits in the inputs, which we keep exactly.

## 4. Honest models on the test month

| Model | AUC | PR-AUC | Log loss | Saved at money threshold | Saved at 0.5 |
|---|---|---|---|---|---|
| Nobody churns (baseline) | 0.500 | 0.119 | 0.373 | €6,255 (t=0.01) | €0 |
| Auto-renew rule | 0.594 | 0.227 | 0.364 | €6,255 (t=0.01) | €1,430 |
| Logistic regression | 0.725 | 0.362 | 0.352 | €9,035 (t=0.03) | €2,560 |
| LightGBM | 0.786 | 0.428 | 0.317 | €9,510 (t=0.06) | €2,680 |

## 5–6. Money

Assumptions: offer costs €5, a lost customer costs €60, offer keeps 100% of the leavers it reaches. Break-even chance of leaving: 8.3%.

| Who gets an offer | Threshold | Offers | Leavers reached | Leavers missed | Saved |
|---|---|---|---|---|---|
| money-based threshold | 0.06 | 846 | 229 | 120 | €9,510 |
| default 0.5 | 0.50 | 76 | 51 | 298 | €2,680 |
| offer to everyone | 0.00 | 2,937 | 349 | 0 | €6,255 |
| no offers | 1.00 | 0 | 0 | 349 | €0 |

Best possible threshold in hindsight: 0.03 (€10,065). Ours was picked a month earlier, without seeing the answers.

Most useful inputs for LightGBM (share of total gain):

- `auto_renew`: 36.0%
- `secs_m1`: 6.9%
- `completion_ratio`: 6.7%
- `unique_songs_m1`: 6.5%
- `secs_m3`: 4.1%
- `tenure_days`: 3.8%
- `reg_days`: 3.8%
- `auto_renew_share`: 3.5%
- `activity_trend`: 3.1%
- `days_to_expire`: 3.1%
