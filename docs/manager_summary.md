# Churn models: what to trust and what to do

*One-page summary for managers. Based on KKBox subscription data, 2015-01-01 to 2017-03-31.*

## The short version

Popular public churn models look far better than they are. When we re-tested three of them the honest way - only using what the company knew on the day of the prediction - their scores fell. The worst one (jsroa15/KKBOX) went from **0.97 to 0.78** on a scale where 0.5 is a coin flip and 1.0 is perfect.

## Why the published scores were wrong

The models were accidentally allowed to peek at the future. For example, they used:
- the number of cancellations, counted to the end of the file
- the membership end date taken after the customer had already renewed
- the date of the customer's latest payment, including payments made after the prediction date

That is like predicting who will cancel a gym membership by checking next month's attendance list. It scores well in a test and is useless on Monday morning.

## What an honest model is really worth

Our own model, trained only on the past and tested on a later month it had never seen:

- Ranks customers by risk with a score of **0.87** (the simple "who turned off auto-renew" rule scores 0.80).
- Per 1,000 customers due to renew, it would save about **€1,159** with the threshold set by money, versus €336 with the usual default setting.

## Who should get an offer

Send an offer when a customer's chance of leaving is above **9%**, not the default 50%. With an offer costing €5 and a lost customer costing €60, an offer pays for itself if even 1 in 12 recipients would otherwise leave. The default setting sends only 7,619 offers and misses 29,291 of 34,878 leavers; the money-based setting sends 81,371 offers and reaches 23,856 of them.

## What to do

1. **Don't trust a churn score unless you know it was tested on a later month than it was built on.** Ask "what date was the cut-off?" before any other question.
2. **Set the offer rule in euros, not in accuracy.** Revisit the €5 / €60 figures with finance - the right threshold moves with them (use the calculator on the monitor page).
3. **Watch the model every month.** Our monitor warns when new customers look different from the ones the model learned from, and retrains when accuracy falls. In the replay, the warning came about two months before the drop in accuracy could even be measured.
4. **Test the offer itself.** These numbers assume the offer keeps every leaver it reaches. Hold back a random 10% control group to measure the real effect.
