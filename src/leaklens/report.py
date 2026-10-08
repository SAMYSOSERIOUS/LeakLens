"""Write the human-readable outputs from reports/audit.json.

  reports/results.md        full tables (for the technical reader)
  docs/manager_summary.md   one page, plain English, what to trust and do
  README.md                 the results block between the RESULTS markers
"""

from __future__ import annotations

import re
from pathlib import Path


def _eur(x: float) -> str:
    return f"€{x:,.0f}"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _demo_banner(r: dict) -> str:
    if r["is_sample"]:
        return ("> **Demo numbers.** These come from the built-in made-up sample data, not from KKBox. "
                "Run the pipeline on the real files (see *Run it on the real data*) to get real numbers.\n\n")
    return ""


def _leak_names(rec: dict, k: int = 3) -> list[str]:
    leaked = [l for l in rec["leaks"] if l["leaked"]
              and l["signal_published"] - l["signal_honest"] > 0.02]
    return [l["feature"] for l in leaked[:k]]


def headline_table(r: dict) -> str:
    rows = ["| Public notebook | What it claimed | Re-run as published (AUC) | Honest re-test (AUC) | Drop | Main leak |",
            "|---|---|---|---|---|---|"]
    for rec in r["recipes"]:
        res = rec["results"]
        leaks = ", ".join(f"`{n}`" for n in _leak_names(rec)) or "-"
        rows.append(
            f"| [{rec['title']}]({rec['url']}) | {rec['claimed_text']} | "
            f"{res['as_published']['auc']:.3f} | {res['honest_time_split']['auc']:.3f} | "
            f"−{rec['auc_drop']:.3f} | {leaks} |")
    return "\n".join(rows)


def results_md(r: dict) -> str:
    m = r["models"]
    out = [f"# Audit results\n\n{_demo_banner(r)}",
           f"Data: {r['data']['first_date']} to {r['data']['last_date']}, "
           f"{r['data']['n_users']:,} customers, {r['data']['n_transactions']:,} transactions.  ",
           f"Test month (cut-off): **{r['cutoffs']['test']}** · threshold tuned on {r['cutoffs']['valid']} · "
           f"trained on {', '.join(r['cutoffs']['train'])}.\n",
           "## 1–3. Public notebooks, re-tested\n", headline_table(r), ""]
    for rec in r["recipes"]:
        res = rec["results"]
        out += [f"### {rec['key']}. {rec['title']}\n", f"Claimed: {rec['claimed_text']}\n"]
        out += ["| Setting | AUC | Accuracy | F1 | Precision | Recall |", "|---|---|---|---|---|---|"]
        names = {"as_published": "As published (whole file, random split)",
                 "clean_features_random_split": "Leak-free features, still random split",
                 "honest_time_split": "Honest (cut-off date, train on past, test on future)"}
        for k, label in names.items():
            s = res[k]
            out.append(f"| {label} | {s['auc']:.3f} | {s['accuracy']:.3f} | {s['f1']:.3f} | "
                       f"{s['precision']:.3f} | {s['recall']:.3f} |")
        out += ["", "Which features changed when the future was removed:\n",
                "| Feature | Customers whose value changed | Signal with future data | Signal without |",
                "|---|---|---|---|"]
        for l in rec["leaks"]:
            if l["leaked"]:
                out.append(f"| `{l['feature']}` | {_pct(l['changed_share'])} | {l['signal_published']:.3f} | "
                           f"{l['signal_honest']:.3f} |")
        out += ["", *[f"- {n}" for n in rec["notes"]], ""]

    out += ["## 4. Honest models on the test month\n",
            "| Model | AUC | PR-AUC | Log loss | Saved at money threshold | Saved at 0.5 |",
            "|---|---|---|---|---|---|"]
    for name, s in m["table"].items():
        out.append(f"| {name} | {s['auc']:.3f} | {s['pr_auc']:.3f} | {s['log_loss']:.3f} | "
                   f"{_eur(s['saved_eur_money_threshold'])} (t={s['money_threshold']:.2f}) | "
                   f"{_eur(s['saved_eur_default_0_5'])} |")
    mo = m["money"]
    a = r["assumptions"]
    out += ["", "## 5–6. Money\n",
            f"Assumptions: offer costs €{a['offer_cost']:.0f}, a lost customer costs €{a['customer_value']:.0f}, "
            f"offer keeps {a['save_rate']:.0%} of the leavers it reaches. "
            f"Break-even chance of leaving: {_pct(mo['break_even_chance'])}.\n",
            "| Who gets an offer | Threshold | Offers | Leavers reached | Leavers missed | Saved |",
            "|---|---|---|---|---|---|"]
    for k, v in mo["comparison"].items():
        out.append(f"| {k} | {v['threshold']:.2f} | {v['offers_sent']:,} | {v['leavers_reached']:,} | "
                   f"{v['leavers_missed']:,} | {_eur(v['saved_eur'])} |")
    out += ["", f"Best possible threshold in hindsight: {mo['hindsight_best_threshold']:.2f} "
                f"({_eur(mo['hindsight_best_saved_eur'])}). Ours was picked a month earlier, without seeing the answers.\n",
            "Most useful inputs for LightGBM (share of total gain):\n"]
    out += [f"- `{k}`: {_pct(v)}" for k, v in m["top_features"].items()]
    return "\n".join(out) + "\n"


def manager_md(r: dict) -> str:
    m, mo, a = r["models"], r["models"]["money"], r["assumptions"]
    gbm = m["table"]["LightGBM"]
    rule = m["table"]["Auto-renew rule"]
    cmp_ = mo["comparison"]
    smart, half = cmp_["money-based threshold"], cmp_["default 0.5"]
    pubs = r["recipes"]
    worst = max(pubs, key=lambda x: x["auc_drop"])
    n = gbm["n"]
    per_1000 = lambda v: v / n * 1000
    leak_words = {
        "membership_expire_date": "the membership end date taken after the customer had already renewed",
        "transaction_date": "the date of the customer's latest payment, including payments made after the prediction date",
        "is_cancel": "whether the customer cancelled - recorded after the prediction date",
        "regist_cancels": "the number of cancellations, counted to the end of the file",
        "is_auto_renew": "the auto-renew setting, read after the prediction date",
        "trans_count": "the number of payments, counted to the end of the file",
        "regist_trans": "the number of payments, counted to the end of the file",
    }
    causes = []
    for rec in pubs:
        for f in _leak_names(rec, 2):
            w = leak_words.get(f)
            if w and w not in causes:
                causes.append(w)

    return f"""# Churn models: what to trust and what to do

*One-page summary for managers. {"Numbers below come from made-up demo data; the real-data version replaces them automatically." if r["is_sample"] else f"Based on KKBox subscription data, {r['data']['first_date']} to {r['data']['last_date']}."}*

## The short version

Popular public churn models look far better than they are. When we re-tested three of them the honest way - only using what the company knew on the day of the prediction - their scores fell. The worst one ({worst['title'].split(' (')[0]}) went from **{worst['results']['as_published']['auc']:.2f} to {worst['results']['honest_time_split']['auc']:.2f}** on a scale where 0.5 is a coin flip and 1.0 is perfect.

## Why the published scores were wrong

The models were accidentally allowed to peek at the future. For example, they used:
{chr(10).join(f"- {c}" for c in causes[:3])}

That is like predicting who will cancel a gym membership by checking next month's attendance list. It scores well in a test and is useless on Monday morning.

## What an honest model is really worth

Our own model, trained only on the past and tested on a later month it had never seen:

- Ranks customers by risk with a score of **{gbm['auc']:.2f}** (the simple "who turned off auto-renew" rule scores {rule['auc']:.2f}).
- Per 1,000 customers due to renew, it would save about **{_eur(per_1000(smart['saved_eur']))}** with the threshold set by money, versus {_eur(per_1000(half['saved_eur']))} with the usual default setting.

## Who should get an offer

Send an offer when a customer's chance of leaving is above **{smart['threshold']:.0%}**, not the default 50%. With an offer costing €{a['offer_cost']:.0f} and a lost customer costing €{a['customer_value']:.0f}, an offer pays for itself if even 1 in {round(1 / mo['break_even_chance'])} recipients would otherwise leave. The default setting sends only {half['offers_sent']:,} offers and misses {half['leavers_missed']:,} of {half['leavers_missed'] + half['leavers_reached']:,} leavers; the money-based setting sends {smart['offers_sent']:,} offers and reaches {smart['leavers_reached']:,} of them.

## What to do

1. **Don't trust a churn score unless you know it was tested on a later month than it was built on.** Ask "what date was the cut-off?" before any other question.
2. **Set the offer rule in euros, not in accuracy.** Revisit the €{a['offer_cost']:.0f} / €{a['customer_value']:.0f} figures with finance - the right threshold moves with them (use the calculator on the monitor page).
3. **Watch the model every month.** Our monitor warns when new customers look different from the ones the model learned from, and retrains when accuracy falls. In the replay, the warning came about two months before the drop in accuracy could even be measured.
4. **Test the offer itself.** These numbers assume the offer keeps every leaver it reaches. Hold back a random 10% control group to measure the real effect.
"""


def write_all(r: dict, root: str | Path = ".") -> None:
    root = Path(root)
    (root / "reports").mkdir(exist_ok=True)
    (root / "docs").mkdir(exist_ok=True)
    (root / "reports" / "results.md").write_text(results_md(r))
    (root / "docs" / "manager_summary.md").write_text(manager_md(r))
    readme = root / "README.md"
    if readme.exists():
        block = (f"<!-- RESULTS:START -->\n{_demo_banner(r)}{headline_table(r)}\n\n"
                 f"Our honest LightGBM on the same test month: AUC **{r['models']['table']['LightGBM']['auc']:.3f}** "
                 f"(auto-renew rule {r['models']['table']['Auto-renew rule']['auc']:.3f}, "
                 f"nobody-churns baseline 0.500). Money-based threshold "
                 f"**{r['models']['money']['threshold_chosen_on_validation']:.2f}** saves "
                 f"**{_eur(r['models']['money']['comparison']['money-based threshold']['saved_eur'])}** on "
                 f"{r['models']['table']['LightGBM']['n']:,} customers, versus "
                 f"{_eur(r['models']['money']['comparison']['default 0.5']['saved_eur'])} at the default 0.5. "
                 f"Full tables: [reports/results.md](reports/results.md).\n<!-- RESULTS:END -->")
        text = re.sub(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: block,
                      readme.read_text(), flags=re.S)
        readme.write_text(text)
    print("wrote reports/results.md, docs/manager_summary.md" + (", README.md" if readme.exists() else ""))
