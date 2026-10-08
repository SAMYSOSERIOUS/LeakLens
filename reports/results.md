# Audit results


Data: 2015-01-01 to 2017-03-31, 2,426,143 customers, 22,975,416 transactions.  
Test month (cut-off): **2017-01-31** · threshold tuned on 2016-11-30 · trained on 2016-07-31, 2016-08-31, 2016-09-30, 2016-10-31.

## 1–3. Public notebooks, re-tested

| Public notebook | What it claimed | Re-run as published (AUC) | Honest re-test (AUC) | Drop | Main leak |
|---|---|---|---|---|---|
| [jsroa15/KKBOX (random forest)](https://github.com/jsroa15/KKBOX) | Random forest, ROC AUC 0.940 on its test set (README). | 0.977 | 0.743 | −0.233 | `mode_quarter`, `regist_cancels`, `revenue` |
| [apostaremczak/churn-prediction (random forest)](https://github.com/apostaremczak/churn-prediction) | 'Unbalanced' random forest: accuracy 92.7%, F1 0.48 (model_results.json). | 0.989 | 0.826 | −0.163 | `membership_expire_date`, `transaction_date`, `is_cancel` |
| [naomifridman/Deep-VAE-prediction-of-churn-customer (VAE + KNN)](https://github.com/naomifridman/Deep-VAE-prediction-of-churn-customer) | KNN on a VAE latent space: accuracy 95.0%, churn-class F1 0.66 (notebook output). | 0.977 | 0.808 | −0.169 | `membership_expire_date`, `transaction_date`, `is_cancel` |

### A. jsroa15/KKBOX (random forest)

Claimed: Random forest, ROC AUC 0.940 on its test set (README).

| Setting | AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| As published (whole file, random split) | 0.977 | 0.967 | 0.617 | 0.567 | 0.676 |
| Leak-free features, still random split | 0.812 | 0.917 | 0.275 | 0.209 | 0.400 |
| Honest (cut-off date, train on past, test on future) | 0.743 | 0.942 | 0.308 | 0.289 | 0.330 |

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
| `mode_payment_method` | 1.1% | 0.740 | 0.740 |

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
| As published (whole file, random split) | 0.977 | 0.974 | 0.581 | 0.818 | 0.450 |
| Leak-free features, still random split | 0.841 | 0.961 | 0.128 | 0.565 | 0.072 |
| Honest (cut-off date, train on past, test on future) | 0.808 | 0.961 | 0.095 | 0.532 | 0.052 |

Which features changed when the future was removed:

| Feature | Customers whose value changed | Signal with future data | Signal without |
|---|---|---|---|
| `membership_expire_date` | 20.0% | 0.952 | 0.519 |
| `transaction_date` | 8.1% | 0.950 | 0.562 |
| `is_cancel` | 3.5% | 0.654 | 0.526 |
| `trans_count` | 97.4% | 0.767 | 0.722 |
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
| Logistic regression | 0.853 | 0.340 | 0.120 | €971,465 (t=0.10) | €152,610 |
| LightGBM | 0.865 | 0.378 | 0.116 | €1,007,035 (t=0.09) | €126,795 |

## 5–6. Money

Assumptions: offer costs €5, a lost customer costs €60, offer keeps 100% of the leavers it reaches. Break-even chance of leaving: 8.3%.

| Who gets an offer | Threshold | Offers | Leavers reached | Leavers missed | Saved |
|---|---|---|---|---|---|
| money-based threshold | 0.09 | 87,085 | 24,041 | 10,837 | €1,007,035 |
| default 0.5 | 0.50 | 3,321 | 2,390 | 32,488 | €126,795 |
| offer to everyone | 0.00 | 883,727 | 34,878 | 0 | −€2,325,955 |
| no offers | 1.00 | 0 | 0 | 34,878 | €0 |

Best possible threshold in hindsight: 0.08 (€1,007,420). Ours was picked a month earlier, without seeing the answers.

Most useful inputs for LightGBM (share of total gain):

- `payment_method`: 40.6%
- `auto_renew`: 16.7%
- `amount_paid`: 10.3%
- `avg_paid`: 6.8%
- `auto_renew_share`: 4.5%
- `cancel_last`: 4.2%
- `n_tx_180d`: 3.5%
- `list_price`: 3.1%
- `n_tx`: 1.7%
- `days_since_last_tx`: 1.6%
