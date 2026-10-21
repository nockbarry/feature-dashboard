#!/usr/bin/env python3
"""
Fraud Feature Explorer v2 -- standalone HTML generator.

Builds a single self-contained HTML file (no CDN, works offline, can be emailed
or dropped on a share) for browsing the features of a transaction fraud model:
English description, exact SQL / Python implementation, univariate performance
over time (AUC / KS / IV / recall and $ recall at a fixed FPR), stability (PSI),
segment breakdowns, in-model behaviour (SHAP dependence, importance over time,
drop-column impact), distributions, missingness, lineage graph, correlation /
redundancy analysis, event annotations and configurable alert rules.

Usage
-----
    # synthetic demo catalog (27 realistic card-fraud features, 21 months)
    python feature_explorer.py --demo -o feature_explorer.html

    # write the demo data as JSON to see / copy the input schema
    python feature_explorer.py --demo --dump-json demo_features.json

    # your own data, optionally with alert-rule overrides
    python feature_explorer.py -i my_features.json -o explorer.html --title "CNP Fraud v7" --rules rules.json

Python stdlib only.

Input schema (JSON)
-------------------
{
  "title": "...",
  "model": {"name": "...", "version": "...", "target": "...",
            "base_fraud_rate": 0.006, "baseline_periods": 3,
            "label_maturity_periods": 2,          # trailing periods whose labels are still arriving
            "fpr_budget": 0.01, "monthly_fraud_value_usd": 3200000,
            "alert_rules": {                      # merged over the built-in defaults; editable in the page
              "psi": {"warning": 0.1, "critical": 0.25},
              "missing_increase": {"warning": 0.05, "critical": 0.15},
              "missing_spike": {"warning": 0.10, "lookback": 6},
              "auc_drop": {"warning": 0.03, "critical": 0.06},
              "recall_drop": {"warning": 0.05, "critical": 0.10},
              "segment_psi": {"warning": 0.2, "critical": 0.35, "min_share": 0.05}}},
  "annotations": [{"date": "2026-01-12", "label": "BNPL onboarding", "text": "..."}],   # global events
  "correlations": {"features": [names...], "matrix": [[1, 0.3, ...], ...]},             # square, symmetric
  "features": [
    {
      "name": "card_txn_count_1h",            # required, unique
      "group": "Velocity", "entity": "card", "dtype": "numeric|categorical|boolean",
      "status": "production|candidate|deprecated", "owner": "...", "version": "1.2.0",
      "created": "2024-03-01", "tags": ["..."], "window": "1h", "freshness": "real-time",
      "default_value": "0", "description": "...", "rationale": "...",
      "source_tables": ["..."], "upstream": ["other_feature"],
      "code": {"sql": "...", "python": "..."},
      "importance": {"shap_mean_abs": 0.12, "gain": 0.08, "rank": 3},
      "summary": {"mean": .., "std": .., "min": .., "p50": .., "max": .., "zero_rate": .., "n_unique": ..},
      "timeline": {
        "periods": ["2025-01", ...],
        "auc": [...], "ks": [...], "iv": [...], "psi": [...],
        "recall_at_1pct_fpr": [...], "value_recall_at_1pct_fpr": [...],
        "missing_rate": [...], "volume": [...], "mean": [...],
        "quantiles": {"p05": [...], "p25": [...], "p50": [...], "p75": [...], "p95": [...]},
        "category_shares": {"cat_a": [...], ...}          # categorical / boolean only
      },
      "segments": {"channel": {"ecom": {"share": [...], "auc": [...], "recall_at_1pct_fpr": [...],
                                        "psi": [...], "missing_rate": [...]}, ...}, "region": {...}},
      "model_behavior": {
        "shap_points": [[x, shap], ...],                  # x numeric or category string
        "shap_missing_mean": -0.01,
        "importance_timeline": [...],                     # mean |SHAP| per period
        "ablation": {"delta_auc": -0.004, "delta_value_recall": -0.02, "usd_at_risk_per_month": 64000,
                     "method": "..."}},
      "distribution": {"bins": ["[0,1)", ..., "Missing"], "baseline": [shares], "current": [shares],
                       "fraud_rate": [rates], "fraud_rate_period": "2026-07"},
      "alert_overrides": {"missing_increase": {"warning": 0.15}, "disabled": ["auc_drop"],
                          "muted": false, "note": "why"},
      "correlations": [{"feature": "...", "r": 0.71}],   # optional; derived from the matrix if absent
      "changelog": [{"date": "2025-06-01", "version": "1.1.0", "note": "..."}]   # also drawn on charts
    }
  ]
}
Every block except "name" is optional; sections without data render a
placeholder. KPIs and alerts are computed in the page from the timelines and
the alert rules (so rules can be tuned live and exported back to JSON).
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import html
import json
import math
import random
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Demo data
# --------------------------------------------------------------------------

PERIODS = [f"{y}-{m:02d}" for y in (2025, 2026) for m in range(1, 13)][:21]  # 2025-01 .. 2026-09
BASE_FRAUD_RATE = 0.006
N_LEGIT, N_FRAUD = 2500, 250


def _poisson(r: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, round(r.normalvariate(lam, math.sqrt(lam))))
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= r.random()
        if p <= L:
            return k
        k += 1


def _nb(r: random.Random, lam: float, k: float = 2.0) -> int:
    """Over-dispersed count (gamma-Poisson)."""
    return _poisson(r, r.gammavariate(k, lam / k))


def _ln(r, mu, s):
    return r.lognormvariate(mu, s)


def _cat(r, probs: dict):
    x, acc = r.random() * sum(probs.values()), 0.0
    for k, p in probs.items():
        acc += p
        if x <= acc:
            return k
    return k


def _mcc_probs(m, fraud):
    if fraud:
        return {"grocery": 6, "fuel": 9, "restaurants": 4, "retail": 18, "travel": 12,
                "digital_goods": 32 + m * 0.6, "gambling": 9, "other": 10}
    return {"grocery": 28, "fuel": 12, "restaurants": 17, "retail": 21, "travel": 5,
            "digital_goods": 7 + m * 0.35, "gambling": 1.5, "other": 8.5}


def _channel_probs(m, fraud):
    if fraud:
        return {"chip": 8, "contactless": 6 + m * 0.3, "swipe": 14, "ecom": 62, "moto": 10}
    return {"chip": 38 - m * 0.6, "contactless": 30 + m * 0.8, "swipe": 7, "ecom": 22, "moto": 3}


def _email_probs(m, fraud):
    if fraud:
        return {"low": 30, "medium": 25, "high": 20, "disposable": 18, "unknown": 7}
    return {"low": 74, "medium": 17, "high": 4, "disposable": 1, "unknown": 4}


def _hour(r, fraud):
    if fraud and r.random() < 0.45:
        return round(r.uniform(0, 5.99), 2)
    return round(r.normalvariate(14, 4.2) % 24, 2)


# Each demo feature: metadata + gen(rng, month_idx, is_fraud) + missing(month_idx) -> rate
DEMO_FEATURES = [
    dict(
        name="txn_amount_usd", group="Amount", entity="transaction", dtype="numeric",
        status="production", owner="fraud-ds", version="1.0.0", created="2023-02-14",
        window="point-in-time", freshness="real-time", default_value="n/a (never null)",
        tags=["core", "monetary"],
        description="Authorized transaction amount converted to USD at the auth-time FX rate.",
        rationale="Fraudsters maximise take before the card is blocked, so amounts skew high; also the base for most ratio features.",
        source_tables=["raw.card_auth_events", "ref.fx_rates_daily"],
        upstream=[],
        sql="""SELECT
  a.auth_id,
  ROUND(a.amount_local * fx.usd_rate, 2) AS txn_amount_usd
FROM raw.card_auth_events a
JOIN ref.fx_rates_daily fx
  ON fx.currency_code = a.currency_code
 AND fx.rate_date     = DATE(a.auth_ts)""",
        python="""def txn_amount_usd(auth: pd.DataFrame, fx: pd.DataFrame) -> pd.Series:
    fx = fx.rename(columns={"rate_date": "auth_date"})
    df = auth.assign(auth_date=auth["auth_ts"].dt.date).merge(
        fx, on=["currency_code", "auth_date"], how="left", validate="m:1"
    )
    return (df["amount_local"] * df["usd_rate"]).round(2).rename("txn_amount_usd")""",
        gen=lambda r, m, f: round(_ln(r, 4.3, 1.3) if f else _ln(r, 3.6 + 0.012 * m, 1.1), 2),
        missing=lambda m: 0.0,
    ),
    dict(
        name="amt_zscore_card_90d", group="Amount", entity="card", dtype="numeric",
        status="production", owner="fraud-ds", version="2.1.0", created="2023-05-02",
        window="90d trailing (excl. current)", freshness="real-time", default_value="NULL (< 5 prior txns)",
        tags=["behavioral", "monetary", "anomaly"],
        description="How many standard deviations the current amount is from the card's own 90-day average amount.",
        rationale="Captures out-of-pattern spend for the specific cardholder, which is far more predictive than raw amount for established cards.",
        source_tables=["feature_store.card_auth_history"],
        upstream=["txn_amount_usd"],
        sql="""SELECT
  auth_id,
  CASE WHEN cnt_90d >= 5 AND std_90d > 0
       THEN (txn_amount_usd - avg_90d) / std_90d
  END AS amt_zscore_card_90d
FROM (
  SELECT
    auth_id, txn_amount_usd,
    AVG(txn_amount_usd)    OVER w AS avg_90d,
    STDDEV(txn_amount_usd) OVER w AS std_90d,
    COUNT(*)               OVER w AS cnt_90d
  FROM feature_store.card_auth_history
  WINDOW w AS (
    PARTITION BY card_id ORDER BY auth_ts
    RANGE BETWEEN INTERVAL '90' DAY PRECEDING AND INTERVAL '1' SECOND PRECEDING
  )
) s""",
        python="""def amt_zscore_card_90d(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values(["card_id", "auth_ts"]).set_index("auth_ts")
    roll = df.groupby("card_id")["txn_amount_usd"].rolling("90D", closed="left")
    stats = pd.DataFrame({"avg": roll.mean(), "std": roll.std(), "cnt": roll.count()})
    stats.index = df.index
    z = (df["txn_amount_usd"].values - stats["avg"]) / stats["std"]
    return z.where((stats["cnt"] >= 5) & (stats["std"] > 0)).rename("amt_zscore_card_90d")""",
        gen=lambda r, m, f: round(r.normalvariate(1.4, 1.6) if f else r.normalvariate(0, 1), 3),
        missing=lambda m: 0.03,
        # BNPL partners onboarded Jan 2026 shift legit e-commerce amounts -> drift hidden in one segment
        seg_gen=lambda r, m, f, ch, rg: round(r.normalvariate(0.9, 1.4), 3) if (ch == "ecom" and not f and m >= 12) else None,
        changelog_extra=[("2026-01-12", "2.1.0", "BNPL merchant onboarding shifted legit e-commerce amount patterns; overall PSI looks fine, ecom segment does not.")],
    ),
    dict(
        name="amt_ratio_card_avg_30d", group="Amount", entity="card", dtype="numeric",
        status="production", owner="fraud-ds", version="1.3.0", created="2023-05-02",
        window="30d trailing (excl. current)", freshness="real-time", default_value="NULL (no prior txns)",
        tags=["behavioral", "monetary"],
        description="Current amount divided by the card's average amount over the previous 30 days.",
        rationale="Scale-free spend spike signal; more robust than the z-score for cards with very few prior transactions.",
        source_tables=["feature_store.card_auth_history"],
        upstream=["txn_amount_usd"],
        sql="""SELECT
  auth_id,
  txn_amount_usd / NULLIF(AVG(txn_amount_usd) OVER (
      PARTITION BY card_id ORDER BY auth_ts
      RANGE BETWEEN INTERVAL '30' DAY PRECEDING AND INTERVAL '1' SECOND PRECEDING
  ), 0) AS amt_ratio_card_avg_30d
FROM feature_store.card_auth_history""",
        python="""def amt_ratio_card_avg_30d(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values(["card_id", "auth_ts"]).set_index("auth_ts")
    avg = (df.groupby("card_id")["txn_amount_usd"]
             .rolling("30D", closed="left").mean().values)
    return (df["txn_amount_usd"] / pd.Series(avg, index=df.index).replace(0, np.nan)) \\
        .rename("amt_ratio_card_avg_30d")""",
        gen=lambda r, m, f: round(_ln(r, 0.7, 0.8) if f else _ln(r, 0.0, 0.5), 3),
        missing=lambda m: 0.04,
    ),
    dict(
        name="is_round_amount", group="Amount", entity="transaction", dtype="boolean",
        status="production", owner="fraud-ds", version="1.0.0", created="2023-02-14",
        window="point-in-time", freshness="real-time", default_value="false",
        tags=["monetary", "heuristic"],
        description="True when the USD amount is a whole multiple of 50 (e.g. 100.00, 250.00).",
        rationale="Card testing and cash-out attempts frequently use round amounts; gift-card and money-transfer fraud especially.",
        source_tables=["raw.card_auth_events"],
        upstream=["txn_amount_usd"],
        sql="""SELECT
  auth_id,
  (MOD(CAST(ROUND(txn_amount_usd * 100) AS BIGINT), 5000) = 0) AS is_round_amount
FROM feature_store.card_auth_history""",
        python="""def is_round_amount(df: pd.DataFrame) -> pd.Series:
    cents = (df["txn_amount_usd"] * 100).round().astype("int64")
    return (cents % 5000 == 0).rename("is_round_amount")""",
        gen=lambda r, m, f: "true" if r.random() < (0.22 if f else 0.08) else "false",
        missing=lambda m: 0.0,
    ),
    dict(
        name="card_txn_count_1h", group="Velocity", entity="card", dtype="numeric",
        status="production", owner="fraud-ds", version="3.0.1", created="2022-11-08",
        window="1h trailing (excl. current)", freshness="real-time (streaming)", default_value="0",
        tags=["core", "velocity", "streaming"],
        description="Number of authorization attempts on the same card in the 60 minutes before this transaction.",
        rationale="Classic burst / card-testing signal. Note: fraud rings have been throttling attempts, so separation is decaying.",
        source_tables=["stream.card_auth_events"],
        upstream=[],
        sql="""SELECT
  auth_id,
  COUNT(*) OVER (
    PARTITION BY card_id ORDER BY auth_ts
    RANGE BETWEEN INTERVAL '1' HOUR PRECEDING AND INTERVAL '1' SECOND PRECEDING
  ) AS card_txn_count_1h
FROM raw.card_auth_events""",
        python="""# Flink / streaming definition mirrored offline for training
def card_txn_count_1h(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values(["card_id", "auth_ts"]).set_index("auth_ts")
    out = (df.assign(one=1).groupby("card_id")["one"]
             .rolling("1h", closed="left").sum().fillna(0).astype("int32"))
    return pd.Series(out.values, index=df["auth_id"], name="card_txn_count_1h")""",
        gen=lambda r, m, f: _nb(r, 2.6 - 0.07 * m) if f else _nb(r, 0.6),
        missing=lambda m: 0.0,
    ),
    dict(
        name="card_txn_count_24h", group="Velocity", entity="card", dtype="numeric",
        status="production", owner="fraud-ds", version="3.0.1", created="2022-11-08",
        window="24h trailing (excl. current)", freshness="real-time (streaming)", default_value="0",
        tags=["velocity", "streaming"],
        description="Number of authorization attempts on the same card in the 24 hours before this transaction.",
        rationale="Longer-horizon velocity; less sensitive to throttled attacks than the 1h count.",
        source_tables=["stream.card_auth_events"],
        upstream=[],
        sql="""SELECT
  auth_id,
  COUNT(*) OVER (
    PARTITION BY card_id ORDER BY auth_ts
    RANGE BETWEEN INTERVAL '24' HOUR PRECEDING AND INTERVAL '1' SECOND PRECEDING
  ) AS card_txn_count_24h
FROM raw.card_auth_events""",
        python="""def card_txn_count_24h(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values(["card_id", "auth_ts"]).set_index("auth_ts")
    out = (df.assign(one=1).groupby("card_id")["one"]
             .rolling("24h", closed="left").sum().fillna(0).astype("int32"))
    return pd.Series(out.values, index=df["auth_id"], name="card_txn_count_24h")""",
        gen=lambda r, m, f: _nb(r, 7) if f else _nb(r, 3),
        # PSD2 step-up retries double-counted in EU from Apr 2026
        seg_gen=lambda r, m, f, ch, rg: _nb(r, 5.5) if (rg == "EU" and not f and m >= 15) else None,
        changelog_extra=[("2026-04-07", "3.0.1", "EU 3-D Secure step-up retries are logged as separate auths; inflates EU counts (DS-1342).")],
        missing=lambda m: 0.0,
    ),
    dict(
        name="card_amt_sum_24h_usd", group="Velocity", entity="card", dtype="numeric",
        status="production", owner="fraud-ds", version="2.0.0", created="2023-01-19",
        window="24h trailing (excl. current)", freshness="real-time (streaming)", default_value="0",
        tags=["velocity", "monetary", "streaming"],
        description="Total USD amount authorized on the card in the previous 24 hours (approved and declined).",
        rationale="Bust-out behaviour: many mid-size purchases in a short period adds up to an unusual daily spend.",
        source_tables=["stream.card_auth_events"],
        upstream=["txn_amount_usd"],
        sql="""SELECT
  auth_id,
  COALESCE(SUM(txn_amount_usd) OVER (
    PARTITION BY card_id ORDER BY auth_ts
    RANGE BETWEEN INTERVAL '24' HOUR PRECEDING AND INTERVAL '1' SECOND PRECEDING
  ), 0) AS card_amt_sum_24h_usd
FROM feature_store.card_auth_history""",
        python="""def card_amt_sum_24h_usd(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values(["card_id", "auth_ts"]).set_index("auth_ts")
    s = (df.groupby("card_id")["txn_amount_usd"]
           .rolling("24h", closed="left").sum().fillna(0.0).values)
    return pd.Series(s, index=df["auth_id"], name="card_amt_sum_24h_usd")""",
        gen=lambda r, m, f: round(_ln(r, 5.5, 1.2) if f else _ln(r, 4.5, 1.0), 2),
        missing=lambda m: 0.0,
    ),
    dict(
        name="distinct_merchants_card_24h", group="Velocity", entity="card", dtype="numeric",
        status="production", owner="fraud-ds", version="1.1.0", created="2023-03-30",
        window="24h trailing (incl. current)", freshness="near-real-time (≤ 2 min)", default_value="1",
        tags=["velocity", "merchant"],
        description="Number of distinct merchants the card transacted with over the last 24 hours, including the current one.",
        rationale="Compromised cards are typically tried at many merchants quickly; genuine customers revisit the same few.",
        source_tables=["feature_store.card_auth_history"],
        upstream=[],
        sql="""SELECT
  cur.auth_id,
  COUNT(DISTINCT prev.merchant_id) AS distinct_merchants_card_24h
FROM feature_store.card_auth_history cur
JOIN feature_store.card_auth_history prev
  ON prev.card_id = cur.card_id
 AND prev.auth_ts >  cur.auth_ts - INTERVAL '24' HOUR
 AND prev.auth_ts <= cur.auth_ts
GROUP BY cur.auth_id""",
        python="""def distinct_merchants_card_24h(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values(["card_id", "auth_ts"])
    out = {}
    for card, g in df.groupby("card_id", sort=False):
        ts, mer, ids = g["auth_ts"].values, g["merchant_id"].values, g["auth_id"].values
        lo = np.searchsorted(ts, ts - np.timedelta64(24, "h"), side="right")
        for i in range(len(g)):
            out[ids[i]] = len(set(mer[lo[i]:i + 1]))
    return pd.Series(out, name="distinct_merchants_card_24h")""",
        gen=lambda r, m, f: 1 + (_nb(r, 3.5) if f else _nb(r, 1.0)),
        missing=lambda m: 0.0,
    ),
    dict(
        name="declined_count_card_24h", group="Velocity", entity="card", dtype="numeric",
        status="production", owner="fraud-ds", version="1.0.2", created="2023-01-19",
        window="24h trailing (excl. current)", freshness="real-time (streaming)", default_value="0",
        tags=["velocity", "declines"],
        description="Number of declined authorizations on the card in the preceding 24 hours (any decline reason).",
        rationale="Failed attempts (wrong CVV, insufficient funds, expired) precede successful fraud during card testing.",
        source_tables=["stream.card_auth_events"],
        upstream=[],
        sql="""SELECT
  auth_id,
  COUNT(*) FILTER (WHERE response_code <> '00') OVER (
    PARTITION BY card_id ORDER BY auth_ts
    RANGE BETWEEN INTERVAL '24' HOUR PRECEDING AND INTERVAL '1' SECOND PRECEDING
  ) AS declined_count_card_24h
FROM raw.card_auth_events""",
        python="""def declined_count_card_24h(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values(["card_id", "auth_ts"]).set_index("auth_ts")
    declined = (df["response_code"] != "00").astype("int32")
    out = (declined.groupby(df["card_id"])
                   .rolling("24h", closed="left").sum().fillna(0).astype("int32"))
    return pd.Series(out.values, index=df["auth_id"], name="declined_count_card_24h")""",
        gen=lambda r, m, f: _nb(r, 1.2, 1.0) if f else _nb(r, 0.1, 0.8),
        missing=lambda m: 0.0,
    ),
    dict(
        name="secs_since_last_txn_card", group="Velocity", entity="card", dtype="numeric",
        status="production", owner="fraud-ds", version="1.0.0", created="2022-11-08",
        window="since previous txn", freshness="real-time (streaming)", default_value="NULL (first txn)",
        tags=["velocity", "recency"],
        description="Seconds elapsed since the previous authorization on the same card.",
        rationale="Very short gaps indicate automated attempts; complements counts without a fixed window.",
        source_tables=["stream.card_auth_events"],
        upstream=[],
        sql="""SELECT
  auth_id,
  EXTRACT(EPOCH FROM auth_ts - LAG(auth_ts) OVER (
    PARTITION BY card_id ORDER BY auth_ts
  )) AS secs_since_last_txn_card
FROM raw.card_auth_events""",
        python="""def secs_since_last_txn_card(df: pd.DataFrame) -> pd.Series:
    df = df.sort_values(["card_id", "auth_ts"])
    gap = df.groupby("card_id")["auth_ts"].diff().dt.total_seconds()
    return pd.Series(gap.values, index=df["auth_id"], name="secs_since_last_txn_card")""",
        gen=lambda r, m, f: round(_ln(r, 8.4, 2.1) if f else _ln(r, 10, 1.5)),
        missing=lambda m: 0.02,
    ),
    dict(
        name="device_age_days", group="Device", entity="device", dtype="numeric",
        status="production", owner="digital-risk", version="2.2.0", created="2023-07-11",
        window="since first seen", freshness="real-time", default_value="NULL (no device fingerprint)",
        tags=["digital", "device", "sdk"],
        description="Days since the device fingerprint was first seen on any account in our network.",
        rationale="Fraudsters rotate emulators and wiped devices; brand-new devices are over-represented in account takeover.",
        source_tables=["digital.device_sessions", "digital.device_first_seen"],
        upstream=[],
        sql="""SELECT
  s.auth_id,
  DATE_DIFF('day', fs.first_seen_ts, s.session_ts) AS device_age_days
FROM digital.device_sessions s
LEFT JOIN digital.device_first_seen fs
  ON fs.device_fp = s.device_fp
WHERE s.device_fp IS NOT NULL""",
        python="""def device_age_days(sessions: pd.DataFrame, first_seen: pd.DataFrame) -> pd.Series:
    df = sessions.merge(first_seen, on="device_fp", how="left")
    age = (df["session_ts"] - df["first_seen_ts"]).dt.days
    return pd.Series(age.values, index=df["auth_id"], name="device_age_days")""",
        gen=lambda r, m, f: round(_ln(r, 3.3, 1.9) if f else _ln(r, 5.0, 1.3), 1),
        missing=lambda m: 0.38 if m in (14, 15) else 0.06,
        changelog_extra=[("2026-03-04", "2.2.0", "Mobile SDK 8.4 rollout broke fingerprint upload on Android 15 (INC-4471); fixed in SDK 8.4.2 on 2026-04-28.")],
    ),
    dict(
        name="is_new_device_for_account", group="Device", entity="account × device", dtype="boolean",
        status="production", owner="digital-risk", version="1.0.0", created="2023-07-11",
        window="account lifetime", freshness="real-time", default_value="false",
        tags=["digital", "device", "ato"],
        description="True if this device fingerprint has never been used by this account before.",
        rationale="Account takeover usually happens from a device the legitimate owner has never used.",
        source_tables=["digital.device_sessions"],
        upstream=[],
        sql="""SELECT
  s.auth_id,
  NOT EXISTS (
    SELECT 1 FROM digital.device_sessions p
    WHERE p.account_id = s.account_id
      AND p.device_fp  = s.device_fp
      AND p.session_ts < s.session_ts
  ) AS is_new_device_for_account
FROM digital.device_sessions s""",
        python="""def is_new_device_for_account(sessions: pd.DataFrame) -> pd.Series:
    s = sessions.sort_values("session_ts")
    first = ~s.duplicated(subset=["account_id", "device_fp"], keep="first")
    return pd.Series(first.values, index=s["auth_id"], name="is_new_device_for_account")""",
        gen=lambda r, m, f: "true" if r.random() < (0.35 if f else 0.05) else "false",
        missing=lambda m: 0.38 if m in (14, 15) else 0.06,
    ),
    dict(
        name="ip_proxy_score", group="Device", entity="ip", dtype="numeric",
        status="production", owner="digital-risk", version="3.0.0", created="2023-09-01",
        window="point-in-time (vendor lookup)", freshness="real-time (vendor API, p99 40ms)", default_value="NULL (lookup timeout)",
        tags=["digital", "vendor", "network"],
        description="Third-party probability (0–1) that the session IP is a proxy, VPN, Tor exit node or hosting provider.",
        rationale="Anonymising infrastructure hides the fraudster's true location.",
        source_tables=["vendor.ip_intel_responses"],
        upstream=[],
        sql="""SELECT
  r.auth_id,
  CAST(JSON_EXTRACT_SCALAR(r.payload, '$.risk.proxy_probability') AS DOUBLE)
    AS ip_proxy_score
FROM vendor.ip_intel_responses r
WHERE r.api_version = 'v2'""",
        python="""def ip_proxy_score(resp: pd.DataFrame) -> pd.Series:
    resp = resp[resp["api_version"] == "v2"]
    score = resp["payload"].map(lambda p: json.loads(p)["risk"].get("proxy_probability"))
    return pd.Series(score.astype(float).values, index=resp["auth_id"], name="ip_proxy_score")""",
        gen=lambda r, m, f: round(
            (r.betavariate(2.6, 3.0) if f else r.betavariate(1.5, 5.0)) if m >= 13
            else (r.betavariate(2.2, 4.5) if f else r.betavariate(1.2, 8.0)), 4),
        missing=lambda m: 0.015,
        changelog_extra=[("2026-02-01", "3.0.0", "Vendor migrated to v2 model; scores recalibrated upward. Retrain pending — see DS-1288.")],
    ),
    dict(
        name="distinct_cards_per_device_7d", group="Device", entity="device", dtype="numeric",
        status="production", owner="digital-risk", version="1.2.0", created="2023-07-11",
        window="7d trailing (incl. current)", freshness="near-real-time (≤ 5 min)", default_value="NULL (no fingerprint)",
        tags=["digital", "device", "linkage"],
        description="Number of distinct cards used from the same device fingerprint over the last 7 days.",
        rationale="One device cycling through many cards is a hallmark of carding operations.",
        source_tables=["digital.device_sessions", "raw.card_auth_events"],
        upstream=[],
        sql="""SELECT
  cur.auth_id,
  COUNT(DISTINCT prev.card_id) AS distinct_cards_per_device_7d
FROM digital.device_sessions cur
JOIN digital.device_sessions prev
  ON prev.device_fp  = cur.device_fp
 AND prev.session_ts >  cur.session_ts - INTERVAL '7' DAY
 AND prev.session_ts <= cur.session_ts
GROUP BY cur.auth_id""",
        python="""def distinct_cards_per_device_7d(s: pd.DataFrame) -> pd.Series:
    s = s.dropna(subset=["device_fp"]).sort_values(["device_fp", "session_ts"])
    out = {}
    for _, g in s.groupby("device_fp", sort=False):
        ts, cards, ids = g["session_ts"].values, g["card_id"].values, g["auth_id"].values
        lo = np.searchsorted(ts, ts - np.timedelta64(7, "D"), side="right")
        for i in range(len(g)):
            out[ids[i]] = len(set(cards[lo[i]:i + 1]))
    return pd.Series(out, name="distinct_cards_per_device_7d")""",
        gen=lambda r, m, f: 1 + (_nb(r, 1.8, 1.0) if f else _nb(r, 0.05, 0.5)),
        missing=lambda m: 0.38 if m in (14, 15) else 0.06,
    ),
    dict(
        name="email_domain_risk_tier", group="Identity", entity="account", dtype="categorical",
        status="deprecated", owner="fraud-ds", version="1.4.0", created="2022-08-15",
        window="static (refreshed weekly)", freshness="daily batch", default_value="'unknown'",
        tags=["identity", "email", "legacy"],
        description="Risk tier of the account's email domain from an internal lookup (low / medium / high / disposable / unknown).",
        rationale="Disposable and newly registered domains correlate with synthetic identities. Superseded by email_risk_score_v2 (vendor).",
        source_tables=["ref.email_domain_tiers", "core.accounts"],
        upstream=[],
        sql="""SELECT
  a.account_id,
  COALESCE(t.tier, 'unknown') AS email_domain_risk_tier
FROM core.accounts a
LEFT JOIN ref.email_domain_tiers t
  ON t.domain = LOWER(SPLIT_PART(a.email, '@', 2))""",
        python="""def email_domain_risk_tier(accounts: pd.DataFrame, tiers: pd.DataFrame) -> pd.Series:
    domain = accounts["email"].str.split("@").str[-1].str.lower()
    tier = domain.map(tiers.set_index("domain")["tier"]).fillna("unknown")
    return pd.Series(tier.values, index=accounts["account_id"], name="email_domain_risk_tier")""",
        gen=lambda r, m, f: _cat(r, _email_probs(m, f)),
        alert_overrides={"muted": True, "note": "Deprecated; alerts muted until removal in v9."},
        missing=lambda m: 0.0,
        changelog_extra=[("2026-06-15", "1.4.0", "Deprecated: replaced by email_risk_score_v2. Scheduled for removal in model v9.")],
    ),
    dict(
        name="billing_shipping_mismatch", group="Identity", entity="transaction", dtype="boolean",
        status="production", owner="fraud-ds", version="1.1.0", created="2023-04-04",
        window="point-in-time", freshness="real-time", default_value="NULL (no shipping address — card-present)",
        tags=["identity", "ecom", "address"],
        description="True when the order's shipping address does not match the card's billing address (normalised postcode + street number).",
        rationale="Reshipping fraud sends goods to drop addresses. Null for card-present transactions by design (structural missingness).",
        source_tables=["ecom.orders", "core.card_billing_address"],
        upstream=[],
        sql="""SELECT
  o.auth_id,
  CASE WHEN o.ship_postcode IS NULL THEN NULL
       ELSE NOT (normalize_postcode(o.ship_postcode) = normalize_postcode(b.postcode)
                 AND o.ship_street_no = b.street_no)
  END AS billing_shipping_mismatch
FROM ecom.orders o
JOIN core.card_billing_address b ON b.card_id = o.card_id""",
        python="""def billing_shipping_mismatch(orders: pd.DataFrame, billing: pd.DataFrame) -> pd.Series:
    df = orders.merge(billing, on="card_id", how="left", suffixes=("_ship", "_bill"))
    same = (df["ship_postcode"].map(normalize_postcode) == df["postcode"].map(normalize_postcode)) \\
         & (df["ship_street_no"] == df["street_no"])
    out = (~same).where(df["ship_postcode"].notna())
    return pd.Series(out.values, index=df["auth_id"], name="billing_shipping_mismatch")""",
        gen=lambda r, m, f: "true" if r.random() < (0.40 if f else 0.10) else "false",
        missing=lambda m: 0.55,
        alert_overrides={"missing_increase": {"warning": 0.15, "critical": 0.30},
                         "note": "≈55% null by design (card-present transactions have no shipping address)."},
    ),
    dict(
        name="geo_distance_km_from_home", group="Geo", entity="card", dtype="numeric",
        status="production", owner="fraud-ds", version="1.2.0", created="2023-06-20",
        window="home = modal city over 180d", freshness="near-real-time", default_value="NULL (unknown merchant geo)",
        tags=["geo", "behavioral"],
        description="Great-circle distance in km between the merchant location and the cardholder's inferred home city.",
        rationale="Card-present fraud with cloned cards happens far from where the genuine cardholder lives.",
        source_tables=["feature_store.card_home_location", "ref.merchant_geo"],
        upstream=[],
        sql="""SELECT
  a.auth_id,
  haversine_km(h.home_lat, h.home_lon, mg.lat, mg.lon) AS geo_distance_km_from_home
FROM raw.card_auth_events a
JOIN feature_store.card_home_location h ON h.card_id = a.card_id
LEFT JOIN ref.merchant_geo mg            ON mg.merchant_id = a.merchant_id""",
        python="""def geo_distance_km_from_home(auth, home, mgeo) -> pd.Series:
    df = auth.merge(home, on="card_id").merge(mgeo, on="merchant_id", how="left")
    lat1, lon1, lat2, lon2 = map(np.radians, [df.home_lat, df.home_lon, df.lat, df.lon])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return pd.Series(6371.0 * 2 * np.arcsin(np.sqrt(a)), index=df["auth_id"],
                     name="geo_distance_km_from_home")""",
        gen=lambda r, m, f: round(_ln(r, 4.5, 1.8) if f else _ln(r, 2.5, 1.4), 1),
        missing=lambda m: 0.08,
    ),
    dict(
        name="is_cross_border", group="Geo", entity="transaction", dtype="boolean",
        status="production", owner="fraud-ds", version="1.0.0", created="2022-08-15",
        window="point-in-time", freshness="real-time", default_value="false",
        tags=["geo", "core"],
        description="True when the merchant country differs from the card issuing country.",
        rationale="Cross-border usage is a strong prior for compromised-card fraud, especially CNP.",
        source_tables=["raw.card_auth_events", "core.cards"],
        upstream=[],
        sql="""SELECT a.auth_id, (a.merchant_country <> c.issuer_country) AS is_cross_border
FROM raw.card_auth_events a
JOIN core.cards c ON c.card_id = a.card_id""",
        python="""def is_cross_border(auth: pd.DataFrame, cards: pd.DataFrame) -> pd.Series:
    df = auth.merge(cards[["card_id", "issuer_country"]], on="card_id", how="left")
    return pd.Series((df["merchant_country"] != df["issuer_country"]).values,
                     index=df["auth_id"], name="is_cross_border")""",
        gen=lambda r, m, f: "true" if r.random() < (0.18 if f else 0.04) else "false",
        missing=lambda m: 0.0,
    ),
    dict(
        name="geo_velocity_kmh", group="Geo", entity="card", dtype="numeric",
        status="production", owner="fraud-ds", version="1.1.0", created="2024-01-09",
        window="vs previous geolocated txn", freshness="real-time", default_value="NULL (no prior geo)",
        tags=["geo", "impossible-travel", "mobile"],
        description="Implied travel speed (km/h) between this and the card's previous geolocated transaction.",
        rationale="'Impossible travel' — speeds above ~900 km/h mean two physical cards / devices in different places.",
        source_tables=["feature_store.card_auth_history", "digital.device_geo"],
        upstream=["geo_distance_km_from_home"],
        sql="""SELECT
  auth_id,
  haversine_km(lat, lon, prev_lat, prev_lon)
    / NULLIF(EXTRACT(EPOCH FROM auth_ts - prev_ts) / 3600.0, 0) AS geo_velocity_kmh
FROM (
  SELECT auth_id, auth_ts, lat, lon,
         LAG(lat)     OVER w AS prev_lat,
         LAG(lon)     OVER w AS prev_lon,
         LAG(auth_ts) OVER w AS prev_ts
  FROM feature_store.card_auth_history
  WHERE lat IS NOT NULL
  WINDOW w AS (PARTITION BY card_id ORDER BY auth_ts)
) s""",
        python="""def geo_velocity_kmh(df: pd.DataFrame) -> pd.Series:
    df = df.dropna(subset=["lat", "lon"]).sort_values(["card_id", "auth_ts"])
    g = df.groupby("card_id")
    km = haversine_km(df["lat"], df["lon"], g["lat"].shift(), g["lon"].shift())
    hours = g["auth_ts"].diff().dt.total_seconds() / 3600.0
    return pd.Series((km / hours.replace(0, np.nan)).values, index=df["auth_id"],
                     name="geo_velocity_kmh")""",
        gen=lambda r, m, f: round(_ln(r, 3.5, 2.2) if f else _ln(r, 1.0, 1.5), 2),
        missing=lambda m: 0.31 if m >= 16 else 0.12,
        changelog_extra=[("2026-05-12", "1.1.0", "iOS 19 approximate-location default reduced device geo coverage; missingness up ~19pp.")],
    ),
    dict(
        name="merchant_category_group", group="Merchant", entity="merchant", dtype="categorical",
        status="production", owner="fraud-ds", version="1.0.0", created="2022-08-15",
        window="static", freshness="daily batch", default_value="'other'",
        tags=["merchant", "core"],
        description="Merchant category code (MCC) rolled up into 8 business-meaningful groups.",
        rationale="Fraud concentrates in resellable goods: digital goods, electronics retail, travel, gambling.",
        source_tables=["raw.card_auth_events", "ref.mcc_groups"],
        upstream=[],
        sql="""SELECT
  a.auth_id,
  COALESCE(g.mcc_group, 'other') AS merchant_category_group
FROM raw.card_auth_events a
LEFT JOIN ref.mcc_groups g ON g.mcc = a.mcc""",
        python="""MCC_GROUPS = load_ref("ref.mcc_groups").set_index("mcc")["mcc_group"]

def merchant_category_group(auth: pd.DataFrame) -> pd.Series:
    return auth["mcc"].map(MCC_GROUPS).fillna("other").rename("merchant_category_group")""",
        gen=lambda r, m, f: _cat(r, _mcc_probs(m, f)),
        missing=lambda m: 0.0,
    ),
    dict(
        name="merchant_fraud_rate_90d", group="Merchant", entity="merchant", dtype="numeric",
        status="production", owner="fraud-ds", version="2.0.0", created="2023-10-02",
        window="90d trailing, 30d label maturity lag", freshness="daily batch", default_value="global prior (Bayesian shrinkage)",
        tags=["merchant", "target-encoding", "lagged-label"],
        description="Shrunk fraud rate at this merchant over 90 days, ending 30 days ago so chargeback labels have matured.",
        rationale="Compromised or colluding merchants concentrate fraud. The 30-day lag prevents label leakage.",
        source_tables=["labels.chargebacks", "raw.card_auth_events"],
        upstream=[],
        sql="""WITH m AS (
  SELECT merchant_id,
         COUNT(*)                                AS n,
         SUM(CASE WHEN cb.auth_id IS NOT NULL THEN 1 ELSE 0 END) AS n_fraud
  FROM raw.card_auth_events a
  LEFT JOIN labels.chargebacks cb ON cb.auth_id = a.auth_id AND cb.reason_group = 'fraud'
  WHERE a.auth_ts >= CURRENT_DATE - INTERVAL '120' DAY
    AND a.auth_ts <  CURRENT_DATE - INTERVAL '30' DAY
  GROUP BY merchant_id
)
SELECT merchant_id,
       (n_fraud + 50 * 0.006) / (n + 50) AS merchant_fraud_rate_90d   -- prior weight 50
FROM m""",
        python="""PRIOR_RATE, PRIOR_N = 0.006, 50

def merchant_fraud_rate_90d(auth: pd.DataFrame, cb: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    win = auth[(auth.auth_ts >= as_of - pd.Timedelta("120D")) & (auth.auth_ts < as_of - pd.Timedelta("30D"))]
    win = win.assign(is_fraud=win["auth_id"].isin(cb.loc[cb.reason_group == "fraud", "auth_id"]))
    agg = win.groupby("merchant_id")["is_fraud"].agg(["sum", "count"])
    return ((agg["sum"] + PRIOR_N * PRIOR_RATE) / (agg["count"] + PRIOR_N)) \\
        .rename("merchant_fraud_rate_90d")""",
        gen=lambda r, m, f: round(r.betavariate(1.2, 140) if f else r.betavariate(1.0, 300), 5),
        missing=lambda m: 0.0,
    ),
    dict(
        name="merchant_age_days", group="Merchant", entity="merchant", dtype="numeric",
        status="candidate", owner="merchant-risk", version="0.3.0", created="2026-04-18",
        window="since onboarding", freshness="daily batch", default_value="NULL (unknown acquirer record)",
        tags=["merchant", "new"],
        description="Days since the merchant first processed a transaction on our network.",
        rationale="Brand-new merchants include shell merchants set up for transaction laundering.",
        source_tables=["ref.merchant_master"],
        upstream=[],
        sql="""SELECT
  a.auth_id,
  DATE_DIFF('day', mm.first_txn_date, DATE(a.auth_ts)) AS merchant_age_days
FROM raw.card_auth_events a
LEFT JOIN ref.merchant_master mm ON mm.merchant_id = a.merchant_id""",
        python="""def merchant_age_days(auth: pd.DataFrame, mm: pd.DataFrame) -> pd.Series:
    df = auth.merge(mm[["merchant_id", "first_txn_date"]], on="merchant_id", how="left")
    return pd.Series((df["auth_ts"].dt.normalize() - df["first_txn_date"]).dt.days.values,
                     index=df["auth_id"], name="merchant_age_days")""",
        gen=lambda r, m, f: round(_ln(r, 5.5, 1.5) if f else _ln(r, 7.0, 1.0)),
        missing=lambda m: 0.05,
    ),
    dict(
        name="channel", group="Transaction", entity="transaction", dtype="categorical",
        status="production", owner="fraud-ds", version="1.0.0", created="2022-08-15",
        window="point-in-time", freshness="real-time", default_value="n/a",
        tags=["core", "channel"],
        description="Transaction channel derived from POS entry mode: chip, contactless, magstripe swipe, e-commerce, or mail/telephone order.",
        rationale="Card-not-present channels (ecom, MOTO) carry the bulk of fraud losses since EMV adoption.",
        source_tables=["raw.card_auth_events"],
        upstream=[],
        sql="""SELECT
  auth_id,
  CASE
    WHEN pos_entry_mode IN ('05', '95')       THEN 'chip'
    WHEN pos_entry_mode IN ('07', '91')       THEN 'contactless'
    WHEN pos_entry_mode IN ('02', '90')       THEN 'swipe'
    WHEN pos_entry_mode = '81' OR eci IS NOT NULL THEN 'ecom'
    ELSE 'moto'
  END AS channel
FROM raw.card_auth_events""",
        python="""ENTRY_MODE = {"05": "chip", "95": "chip", "07": "contactless", "91": "contactless",
              "02": "swipe", "90": "swipe", "81": "ecom"}

def channel(auth: pd.DataFrame) -> pd.Series:
    ch = auth["pos_entry_mode"].map(ENTRY_MODE)
    ch = ch.mask(ch.isna() & auth["eci"].notna(), "ecom").fillna("moto")
    return ch.rename("channel")""",
        gen=lambda r, m, f: _cat(r, _channel_probs(m, f)),
        from_segment="channel",
        missing=lambda m: 0.0,
    ),
    dict(
        name="hour_of_day_local", group="Transaction", entity="transaction", dtype="numeric",
        status="candidate", owner="fraud-ds", version="0.1.0", created="2026-07-01",
        window="point-in-time", freshness="real-time", default_value="NULL (unknown merchant tz)",
        tags=["temporal", "non-monotonic", "new"],
        description="Hour of day (0–24, fractional) in the merchant's local timezone.",
        rationale="Fraud spikes in the small hours. Relationship is non-monotonic, so univariate AUC understates its value to tree models — check IV and the fraud-rate-by-bin chart.",
        source_tables=["raw.card_auth_events", "ref.merchant_geo"],
        upstream=[],
        sql="""SELECT
  a.auth_id,
  EXTRACT(HOUR   FROM a.auth_ts AT TIME ZONE mg.tz)
  + EXTRACT(MINUTE FROM a.auth_ts AT TIME ZONE mg.tz) / 60.0 AS hour_of_day_local
FROM raw.card_auth_events a
LEFT JOIN ref.merchant_geo mg ON mg.merchant_id = a.merchant_id""",
        python="""def hour_of_day_local(auth: pd.DataFrame, mgeo: pd.DataFrame) -> pd.Series:
    df = auth.merge(mgeo[["merchant_id", "tz"]], on="merchant_id", how="left")
    local = [ts.tz_convert(tz) if isinstance(tz, str) else pd.NaT
             for ts, tz in zip(df["auth_ts"], df["tz"])]
    local = pd.Series(local)
    return pd.Series((local.dt.hour + local.dt.minute / 60.0).values,
                     index=df["auth_id"], name="hour_of_day_local")""",
        gen=lambda r, m, f: _hour(r, f),
        alert_overrides={"auc_drop": {"warning": 0.05, "critical": 0.10},
                         "note": "Non-monotonic feature: univariate AUC is noisy, judge on IV / SHAP instead."},
        missing=lambda m: 0.02,
    ),
    dict(
        name="account_age_days", group="Account", entity="account", dtype="numeric",
        status="production", owner="fraud-ds", version="1.0.0", created="2022-08-15",
        window="since account open", freshness="daily batch", default_value="n/a",
        tags=["account", "core"],
        description="Days between account opening and the transaction date.",
        rationale="New accounts are disproportionately synthetic identities or first-party bust-outs.",
        source_tables=["core.accounts"],
        upstream=[],
        sql="""SELECT a.auth_id, DATE_DIFF('day', acc.opened_date, DATE(a.auth_ts)) AS account_age_days
FROM raw.card_auth_events a
JOIN core.cards    c   ON c.card_id = a.card_id
JOIN core.accounts acc ON acc.account_id = c.account_id""",
        python="""def account_age_days(auth, cards, accounts) -> pd.Series:
    df = auth.merge(cards[["card_id", "account_id"]], on="card_id") \\
             .merge(accounts[["account_id", "opened_date"]], on="account_id")
    return pd.Series((df["auth_ts"].dt.normalize() - df["opened_date"]).dt.days.values,
                     index=df["auth_id"], name="account_age_days")""",
        gen=lambda r, m, f: round(_ln(r, 5.0, 1.6) if f else _ln(r, 6.5, 1.0)),
        missing=lambda m: 0.0,
    ),
    dict(
        name="days_since_credential_change", group="Account", entity="account", dtype="numeric",
        status="production", owner="digital-risk", version="1.1.0", created="2024-02-12",
        window="since last change", freshness="near-real-time", default_value="NULL (never changed)",
        tags=["account", "ato"],
        description="Days since the most recent password, phone number or email change on the account.",
        rationale="Takeover typically involves changing credentials right before spending, to lock the real owner out.",
        source_tables=["digital.account_security_events"],
        upstream=[],
        sql="""SELECT
  a.auth_id,
  DATE_DIFF('day', MAX(e.event_ts), a.auth_ts) AS days_since_credential_change
FROM raw.card_auth_events a
JOIN core.cards c ON c.card_id = a.card_id
LEFT JOIN digital.account_security_events e
  ON e.account_id = c.account_id
 AND e.event_type IN ('password_change', 'phone_change', 'email_change')
 AND e.event_ts < a.auth_ts
GROUP BY a.auth_id, a.auth_ts""",
        python="""CRED_EVENTS = {"password_change", "phone_change", "email_change"}

def days_since_credential_change(auth, cards, events) -> pd.Series:
    ev = events[events["event_type"].isin(CRED_EVENTS)].sort_values("event_ts")
    df = auth.merge(cards[["card_id", "account_id"]], on="card_id").sort_values("auth_ts")
    df = pd.merge_asof(df, ev[["account_id", "event_ts"]], left_on="auth_ts",
                       right_on="event_ts", by="account_id", allow_exact_matches=False)
    return pd.Series((df["auth_ts"] - df["event_ts"]).dt.days.values,
                     index=df["auth_id"], name="days_since_credential_change")""",
        gen=lambda r, m, f: round(_ln(r, 3.4, 1.9) if f else _ln(r, 5.0, 1.2)),
        missing=lambda m: 0.20,
    ),
    dict(
        name="card_shared_ip_count_30d", group="Network", entity="card × ip", dtype="numeric",
        status="production", owner="digital-risk", version="1.0.0", created="2024-05-20",
        window="30d trailing", freshness="hourly batch", default_value="0",
        tags=["network", "linkage", "graph"],
        description="Number of other cards that used the same IP address as this transaction in the last 30 days.",
        rationale="Graph signal for fraud rings: many unrelated cards behind one IP.",
        source_tables=["digital.device_sessions"],
        upstream=[],
        sql="""SELECT
  cur.auth_id,
  COUNT(DISTINCT prev.card_id) FILTER (WHERE prev.card_id <> cur.card_id)
    AS card_shared_ip_count_30d
FROM digital.device_sessions cur
LEFT JOIN digital.device_sessions prev
  ON prev.ip_address = cur.ip_address
 AND prev.session_ts BETWEEN cur.session_ts - INTERVAL '30' DAY AND cur.session_ts
GROUP BY cur.auth_id""",
        python="""def card_shared_ip_count_30d(s: pd.DataFrame) -> pd.Series:
    s = s.sort_values(["ip_address", "session_ts"])
    out = {}
    for _, g in s.groupby("ip_address", sort=False):
        ts, cards, ids = g["session_ts"].values, g["card_id"].values, g["auth_id"].values
        lo = np.searchsorted(ts, ts - np.timedelta64(30, "D"), side="left")
        for i in range(len(g)):
            out[ids[i]] = len(set(cards[lo[i]:i + 1]) - {cards[i]})
    return pd.Series(out, name="card_shared_ip_count_30d")""",
        gen=lambda r, m, f: _nb(r, 2.0, 0.8) if f else _nb(r, 0.3, 0.6),
        missing=lambda m: 0.02,
    ),
]


# --------------------------------------------------------------------------
# Demo simulation settings
# --------------------------------------------------------------------------

REGION_PROBS = {False: {"NA": 55, "EU": 30, "APAC": 15}, True: {"NA": 45, "EU": 25, "APAC": 30}}
SEGMENT_DIMS = {"channel": ["chip", "contactless", "swipe", "ecom", "moto"], "region": ["NA", "EU", "APAC"]}
LABEL_MATURITY = [0.55, 0.85]  # share of eventual fraud labels known in the latest / 2nd-latest period
FPR_BUDGET = 0.01
MEAN_FRAUD_AMOUNT = math.exp(4.3 + 1.3 ** 2 / 2)

DEFAULT_ALERT_RULES = {
    "psi": {"warning": 0.10, "critical": 0.25},
    "missing_increase": {"warning": 0.05, "critical": 0.15},
    "missing_spike": {"warning": 0.10, "lookback": 6},
    "auc_drop": {"warning": 0.03, "critical": 0.06},
    "recall_drop": {"warning": 0.05, "critical": 0.10},
    "segment_psi": {"warning": 0.20, "critical": 0.35, "min_share": 0.05},
}

DEMO_ANNOTATIONS = [
    dict(date="2025-07-01", label="Model v8.0", text="card_txn_fraud_gbm v8.0 deployed (retrained on 2024-H2 data)."),
    dict(date="2025-11-28", label="Black Friday", text="Peak season: volume +35%, e-commerce mix shift."),
    dict(date="2026-01-12", label="BNPL onboarding", text="New buy-now-pay-later merchant partners onboarded in e-commerce."),
    dict(date="2026-06-01", label="Model v8.3", text="v8.3 hotfix: score recalibration only, same features."),
]

# deliberately redundant / related pairs layered on top of the group factor model
DEMO_CORR = {
    ("amt_zscore_card_90d", "amt_ratio_card_avg_30d"): 0.88,
    ("card_txn_count_24h", "card_amt_sum_24h_usd"): 0.84,
    ("card_txn_count_1h", "card_txn_count_24h"): 0.81,
    ("device_age_days", "is_new_device_for_account"): -0.83,
    ("distinct_cards_per_device_7d", "card_shared_ip_count_30d"): 0.79,
    ("geo_distance_km_from_home", "is_cross_border"): 0.64,
    ("geo_velocity_kmh", "geo_distance_km_from_home"): 0.58,
    ("ip_proxy_score", "card_shared_ip_count_30d"): 0.41,
    ("account_age_days", "days_since_credential_change"): 0.37,
    ("merchant_category_group", "channel"): 0.46,
}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _auc(pos: list, neg: list) -> float | None:
    """Mann-Whitney AUC with tie handling."""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i, n = 0.0, 0, len(allv)
    while i < n:
        j = i
        while j < n and allv[j][0] == allv[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        rank_sum += avg_rank * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _ks(pos: list, neg: list) -> float | None:
    if not pos or not neg:
        return None
    pos, neg = sorted(pos), sorted(neg)
    best = 0.0
    for v in sorted(set(pos) | set(neg)):
        best = max(best, abs(bisect.bisect_right(pos, v) / len(pos) - bisect.bisect_right(neg, v) / len(neg)))
    return best


def _recall_at_fpr(pos: list, neg: list, budget: float = FPR_BUDGET):
    """pos: [(score | None, amount)], neg: [score | None].
    Flags score > t, with t chosen so at most `budget` of legit traffic is flagged
    (nulls are never flagged). Returns (count recall, dollar recall)."""
    if not pos or not neg:
        return None, None
    negs = sorted((s for s in neg if s is not None), reverse=True)
    k = int(budget * len(neg))
    t = negs[k] if k < len(negs) else -math.inf
    hit = [s is not None and s > t for s, _ in pos]
    total = sum(a for _, a in pos) or 1.0
    return sum(hit) / len(pos), sum(a for (_, a), h in zip(pos, hit) if h) / total


def _quantile(sorted_vals: list, q: float):
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _psi(base: list, cur: list, eps: float = 1e-4) -> float:
    return sum((c - b) * math.log((c + eps) / (b + eps)) for b, c in zip(base, cur))


def _iv(legit_share: list, fraud_share: list, eps: float = 1e-4) -> float:
    return sum((f - l) * math.log((f + eps) / (l + eps)) for l, f in zip(legit_share, fraud_share))


def _logit(p: float) -> float:
    p = min(max(p, 1e-5), 1 - 1e-5)
    return math.log(p / (1 - p))


def _fmt_edge(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2g}" if v == int(v) else f"{v:.1f}"
    return f"{v:.3g}"


def _r(x, nd=4):
    return None if x is None else round(x, nd)


def _sample(spec: dict, r: random.Random, m: int, fraud: bool):
    """One simulated transaction: (feature value | None, channel, region, amount_usd)."""
    ch = _cat(r, _channel_probs(m, fraud))
    rg = _cat(r, REGION_PROBS[fraud])
    miss = spec["missing"](m)
    if fraud:
        miss = min(1.0, miss * 1.3)  # digital signals are evaded / blocked more often by fraudsters
    if r.random() < miss:
        v = None
    elif spec.get("from_segment") == "channel":
        v = ch
    else:
        v = spec["seg_gen"](r, m, fraud, ch, rg) if "seg_gen" in spec else None
        if v is None:
            v = spec["gen"](r, m, fraud)
    if spec["name"] == "txn_amount_usd" and v is not None:
        amt = v
    else:
        amt = _ln(r, 4.3, 1.3) if fraud else _ln(r, 3.6, 1.1)
    return v, ch, rg, amt


def _corr_matrix(feats: list, rng: random.Random) -> list:
    """Synthetic but coherent correlation matrix: group factor model + known related pairs."""
    groups = sorted({f["group"] for f in feats})
    k = len(groups) + 1
    load, sign = {}, {}
    for f in feats:
        vec = [rng.gauss(0, 0.16) for _ in range(k)]
        vec[groups.index(f["group"])] += 0.75
        vec[-1] += 0.3
        load[f["name"]] = vec
        sign[f["name"]] = -1 if rng.random() < 0.15 else 1
    names = [f["name"] for f in feats]
    M = [[1.0] * len(names) for _ in names]
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            va, vb = load[a], load[b]
            r = sum(x * y for x, y in zip(va, vb)) / math.sqrt(sum(x * x for x in va) * sum(y * y for y in vb))
            r = r * 0.62 * sign[a] * sign[b]
            r = DEMO_CORR.get((a, b), DEMO_CORR.get((b, a), r))
            M[i][j] = M[j][i] = round(max(-0.97, min(0.97, r)), 3)
    return M


def build_demo(seed: int = 7) -> dict:
    rng = random.Random(seed)
    n_base, n_per = 3, len(PERIODS)
    mature = n_per - 1 - len(LABEL_MATURITY)       # last period with complete labels
    feats, deps = [], {}
    w_fraud = BASE_FRAUD_RATE / (1 - BASE_FRAUD_RATE) * N_LEGIT / N_FRAUD  # reweight sampled fraud
    n_real = round(N_LEGIT * BASE_FRAUD_RATE / (1 - BASE_FRAUD_RATE))      # fraud count at the true base rate
    seg_idx = {"channel": 1, "region": 2}

    for spec in DEMO_FEATURES:
        dtype = spec["dtype"]
        months = [([_sample(spec, rng, m, False) for _ in range(N_LEGIT)],
                   [_sample(spec, rng, m, True) for _ in range(N_FRAUD)]) for m in range(n_per)]

        # ---- binning, defined once on the baseline window
        if dtype == "numeric":
            pool = sorted(t[0] for lg, fr in months[:n_base] for t in lg if t[0] is not None)
            uniq = sorted(set(pool))
            if len(uniq) <= 12 and all(float(v).is_integer() for v in uniq):
                # low-cardinality counts: one bin per value, open-ended top bin
                edges = uniq[1:10]
                bins = [_fmt_edge(v) for v in uniq[:len(edges)]] + [f"≥ {_fmt_edge(uniq[len(edges)])}"]
            else:
                edges = sorted(set(_quantile(pool, q / 10) for q in range(1, 10)))
                bins = []
                lo_lbl = _fmt_edge(pool[0])
                for e in edges:
                    bins.append(f"[{lo_lbl}, {_fmt_edge(e)})")
                    lo_lbl = _fmt_edge(e)
                bins.append(f"≥ {lo_lbl}")

            def bin_of(v, edges=edges):
                return bisect.bisect_right(edges, v)
        else:
            cats = {}
            for lg, fr in months[:n_base]:
                for t in lg:
                    if t[0] is not None:
                        cats[t[0]] = cats.get(t[0], 0) + 1
            for lg, fr in months:
                for t in lg + fr:
                    if t[0] is not None and t[0] not in cats:
                        cats[t[0]] = 0
            bins = sorted(cats, key=lambda c: -cats[c])
            idx = {c: i for i, c in enumerate(bins)}

            def bin_of(v, idx=idx):
                return idx[v]
        has_missing_bin = any(t[0] is None for lg, fr in months for t in lg)
        nb = len(bins) + (1 if has_missing_bin else 0)
        labels = bins + (["Missing"] if has_missing_bin else [])

        def bidx(v):
            return nb - 1 if v is None else bin_of(v)

        def hist(rows):
            h = [0] * nb
            for t in rows:
                h[bidx(t[0])] += 1
            tot = sum(h) or 1
            return [c / tot for c in h]

        def bin_rates(lg, fr):
            cl, cf = [0] * nb, [0] * nb
            for t in lg:
                cl[bidx(t[0])] += 1
            for t in fr:
                cf[bidx(t[0])] += 1
            return [(cf[i] * w_fraud) / (cf[i] * w_fraud + cl[i]) if cl[i] + cf[i] else None for i in range(nb)]

        # ---- score orientation: numeric uses the value (sign-flipped if inversely related),
        #      categorical/boolean uses the baseline fraud rate of the category (target encoding)
        if dtype != "numeric":
            hl, hf = [0] * nb, [0] * nb
            for lg, fr in months[:n_base]:
                for t in lg:
                    hl[bidx(t[0])] += 1
                for t in fr:
                    hf[bidx(t[0])] += 1
            enc = [(hf[i] * w_fraud) / (hf[i] * w_fraud + hl[i]) if hl[i] + hf[i] else 0 for i in range(nb)]

            def score(v):
                return None if v is None else enc[bin_of(v)]
        else:
            lg0, fr0 = months[0]
            a0 = _auc([t[0] for t in fr0 if t[0] is not None], [t[0] for t in lg0 if t[0] is not None])
            sgn = -1 if a0 is not None and a0 < 0.5 else 1

            def score(v, sgn=sgn):
                return None if v is None else v * sgn

        def labelled(m):
            """Fraud that has been reported by period m's cut-off; the rest still looks legit."""
            lg, fr = months[m]
            k = n_per - 1 - m
            frac = LABEL_MATURITY[k] if k < len(LABEL_MATURITY) else 1.0
            nl = round(frac * len(fr))
            return fr[:nl], lg + fr[nl:]

        def label_metrics(pos, neg):
            ps = [score(t[0]) for t in pos]
            ns = [score(t[0]) for t in neg]
            pn, nn_ = [s for s in ps if s is not None], [s for s in ns if s is not None]
            rec, vrec = _recall_at_fpr(list(zip(ps, [t[3] for t in pos])), ns)
            return _auc(pn, nn_), _ks(pn, nn_), rec, vrec

        def population(m):
            lg, fr = months[m]
            return lg + fr[:n_real]

        base_hist = [sum(col) / n_base for col in zip(*(hist(population(m)) for m in range(n_base)))]
        seg_base = {}
        for dim, vals in SEGMENT_DIMS.items():
            for val in vals:
                hs = [hist([t for t in population(m) if t[seg_idx[dim]] == val]) for m in range(n_base)]
                seg_base[(dim, val)] = [sum(col) / n_base for col in zip(*hs)]

        tl = dict(periods=PERIODS, auc=[], ks=[], iv=[], recall_at_1pct_fpr=[], value_recall_at_1pct_fpr=[],
                  psi=[], missing_rate=[], volume=[], mean=[])
        if dtype == "numeric":
            tl["quantiles"] = {k: [] for k in ("p05", "p25", "p50", "p75", "p95")}
        else:
            tl["category_shares"] = {c: [] for c in bins}
        segs = {dim: {val: dict(share=[], auc=[], recall_at_1pct_fpr=[], psi=[], missing_rate=[]) for val in vals}
                for dim, vals in SEGMENT_DIMS.items()}

        for m in range(n_per):
            pos, neg = labelled(m)
            auc, ks, rec, vrec = label_metrics(pos, neg)
            pop = population(m)
            tl["auc"].append(_r(auc))
            tl["ks"].append(_r(ks))
            tl["iv"].append(_r(_iv(hist(neg), hist(pos))))
            tl["recall_at_1pct_fpr"].append(_r(rec))
            tl["value_recall_at_1pct_fpr"].append(_r(vrec))
            tl["psi"].append(_r(_psi(base_hist, hist(pop)), 5))
            tl["missing_rate"].append(_r(sum(t[0] is None for t in pop) / len(pop)))
            tl["volume"].append(int(rng.uniform(0.92, 1.08) * (3.1e6 + m * 4.5e4)))
            nn = sorted(t[0] for t in pop if t[0] is not None)
            if dtype == "numeric":
                tl["mean"].append(_r(sum(nn) / len(nn)) if nn else None)
                for k, q in (("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95)):
                    tl["quantiles"][k].append(_r(_quantile(nn, q)))
            else:
                tl["mean"].append(None)
                for c in bins:
                    tl["category_shares"][c].append(_r(sum(1 for v in nn if v == c) / max(1, len(nn))))
            for dim, vals in SEGMENT_DIMS.items():
                si = seg_idx[dim]
                for val in vals:
                    sp = [t for t in pop if t[si] == val]
                    s_auc, _, s_rec, _ = label_metrics([t for t in pos if t[si] == val], [t for t in neg if t[si] == val])
                    d = segs[dim][val]
                    d["share"].append(_r(len(sp) / len(pop)))
                    d["auc"].append(_r(s_auc))
                    d["recall_at_1pct_fpr"].append(_r(s_rec))
                    d["psi"].append(_r(_psi(seg_base[(dim, val)], hist(sp)), 5) if sp else None)
                    d["missing_rate"].append(_r(sum(t[0] is None for t in sp) / len(sp)) if sp else None)

        # ---- latest-period distribution; fraud rate per bin from the last period with mature labels
        cur_hist = hist(population(n_per - 1))
        rates = bin_rates(*months[mature])
        nn = sorted(t[0] for t in population(n_per - 1) if t[0] is not None)
        if dtype == "numeric":
            mean = sum(nn) / len(nn)
            std = math.sqrt(sum((v - mean) ** 2 for v in nn) / len(nn))
            summary = dict(mean=_r(mean), std=_r(std), min=_r(nn[0]), p50=_r(_quantile(nn, .5)),
                           max=_r(nn[-1]), zero_rate=_r(sum(v == 0 for v in nn) / len(nn)),
                           n_unique=len(set(nn)))
        else:
            mode = max(set(nn), key=nn.count)
            summary = dict(n_unique=len(set(nn)), mode=mode, mode_share=_r(nn.count(mode) / len(nn)))

        # ---- raw material for the SHAP dependence plot (tree-model-like step function of bin log-odds)
        lgm, frm = months[mature]
        sample_pts = [t[0] for t in lgm[:240] if t[0] is not None] + [t[0] for t in frm[:60] if t[0] is not None]
        lo_base = _logit(BASE_FRAUD_RATE)
        deps[spec["name"]] = [(v, _logit(rates[bin_of(v)] or BASE_FRAUD_RATE) - lo_base + rng.gauss(0, 0.22))
                              for v in sample_pts]
        miss_rate = rates[nb - 1] if has_missing_bin else None

        changelog = [(spec["created"], "1.0.0" if spec["version"] >= "1" else "0.1.0", "Initial definition.")]
        changelog += spec.get("changelog_extra", [])
        if spec["version"] not in [c[1] for c in changelog]:
            changelog.append(("2025-09-15", spec["version"], "Backfilled with corrected late-arriving events; no logic change."))

        feat = dict(
            name=spec["name"], group=spec["group"], entity=spec["entity"], dtype=dtype,
            status=spec["status"], owner=spec["owner"], version=spec["version"], created=spec["created"],
            tags=spec["tags"], window=spec["window"], freshness=spec["freshness"],
            default_value=spec["default_value"], description=spec["description"],
            rationale=spec["rationale"], source_tables=spec["source_tables"], upstream=spec["upstream"],
            code=dict(sql=spec["sql"], python=spec["python"]),
            summary=summary, timeline=tl, segments=segs,
            distribution=dict(bins=labels, baseline=[_r(x) for x in base_hist],
                              current=[_r(x) for x in cur_hist], fraud_rate=[_r(x, 5) for x in rates],
                              fraud_rate_period=PERIODS[mature]),
            changelog=[dict(date=d, version=v, note=n) for d, v, n in sorted(changelog)],
            _missing_dev=None if not miss_rate else _logit(miss_rate) - lo_base,
        )
        if spec.get("alert_overrides"):
            feat["alert_overrides"] = spec["alert_overrides"]
        feats.append(feat)

    # ---- importance (synthetic, roughly follows univariate strength)
    for f in feats:
        tl = f["timeline"]
        iv = sum(tl["iv"][mature - 2:mature + 1]) / 3
        auc = sum(tl["auc"][mature - 2:mature + 1]) / 3
        f["importance"] = dict(shap_mean_abs=_r(max(0.002, (auc - 0.5) * 0.35 + iv * 0.04) * rng.uniform(0.6, 1.4)),
                               gain=_r(max(0.001, iv * 0.05) * rng.uniform(0.5, 1.5)))
        if f["status"] == "deprecated":
            f["importance"]["shap_mean_abs"] = _r(f["importance"]["shap_mean_abs"] * 0.25)
    for rank, f in enumerate(sorted(feats, key=lambda f: -f["importance"]["shap_mean_abs"]), 1):
        f["importance"]["rank"] = rank

    matrix = _corr_matrix(feats, rng)
    monthly_fraud_value = feats[0]["timeline"]["volume"][-1] * BASE_FRAUD_RATE * MEAN_FRAUD_AMOUNT

    # ---- in-model behaviour: SHAP dependence sample, importance over time, drop-column ablation
    for i, f in enumerate(feats):
        shap = f["importance"]["shap_mean_abs"]
        tl = f["timeline"]
        raw = deps[f["name"]]
        scale = shap / (sum(abs(d) for _, d in raw) / len(raw) or 1)
        iv_base = max(0.05, sum(tl["iv"][:n_base]) / n_base)
        imp_tl = []
        for m in range(n_per):
            iv_m = tl["iv"][min(m, mature)]  # importance needs no labels; reuse last mature IV for recent periods
            drift = 1 + min(1.5, tl["psi"][m]) * 0.4
            imp_tl.append(_r(shap * math.sqrt(max(0.05, iv_m) / iv_base) * drift * rng.uniform(0.96, 1.04), 5))
        max_r = max(abs(matrix[i][j]) for j in range(len(feats)) if j != i)
        d_auc = -shap * 0.05 * (1 - 0.75 * max_r) * rng.uniform(0.7, 1.3)
        d_vrec = d_auc * 5.5
        f["model_behavior"] = dict(
            shap_points=[[x if isinstance(x, str) else _r(x), _r(d * scale, 5)] for x, d in raw],
            shap_missing_mean=None if f["_missing_dev"] is None else _r(f["_missing_dev"] * scale, 5),
            importance_timeline=imp_tl,
            ablation=dict(delta_auc=_r(d_auc, 5), delta_value_recall=_r(d_vrec, 5),
                          usd_at_risk_per_month=round(-d_vrec * monthly_fraud_value),
                          method="retrain without the feature, evaluated on " + PERIODS[mature]),
        )
        del f["_missing_dev"]

    return dict(
        title="Card Fraud Model — Feature Explorer",
        model=dict(name="card_txn_fraud_gbm", version="v8.3", target="is_fraud (confirmed chargeback ≤ 60d)",
                   base_fraud_rate=BASE_FRAUD_RATE, baseline_periods=n_base,
                   label_maturity_periods=len(LABEL_MATURITY), fpr_budget=FPR_BUDGET,
                   monthly_fraud_value_usd=round(monthly_fraud_value),
                   alert_rules=DEFAULT_ALERT_RULES,
                   note="Synthetic demo data. Performance = univariate AUC/KS/IV/recall per month; "
                        "PSI vs. baseline window (first 3 months), decile bins + missing bin. "
                        "The last 2 months have immature chargeback labels."),
        annotations=DEMO_ANNOTATIONS,
        correlations=dict(features=[f["name"] for f in feats], matrix=matrix, method="pearson, latest quarter"),
        features=feats,
    )


# --------------------------------------------------------------------------
# Validation / defaults (alerts and KPIs are computed in the page so rules can be edited live)
# --------------------------------------------------------------------------

def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def derive(data: dict, rules: dict | None = None) -> dict:
    model = data.setdefault("model", {})
    model.setdefault("baseline_periods", 3)
    model.setdefault("label_maturity_periods", 0)
    model.setdefault("fpr_budget", FPR_BUDGET)
    model["alert_rules"] = _merge(_merge(DEFAULT_ALERT_RULES, model.get("alert_rules") or {}), rules or {})
    names = set()
    for f in data.get("features", []):
        if "name" not in f:
            raise ValueError("every feature needs a 'name'")
        if f["name"] in names:
            raise ValueError(f"duplicate feature name: {f['name']}")
        names.add(f["name"])
        f.setdefault("group", "Ungrouped")
        f.setdefault("status", "production")
        f.setdefault("dtype", "numeric")
        f.setdefault("tags", [])
        f.pop("alerts", None)
        f.pop("kpis", None)
    corr = data.get("correlations")
    if corr:
        n = len(corr.get("features", []))
        if len(corr.get("matrix", [])) != n or any(len(row) != n for row in corr["matrix"]):
            raise ValueError("correlations.matrix must be square and match correlations.features")
    data.setdefault("annotations", [])
    data.setdefault("generated_at", dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    return data


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

def render_html(data: dict) -> str:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(data.get("title") or "Feature Explorer")
    return TEMPLATE.replace("__TITLE__", title).replace("__DATA__", payload)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("-i", "--input", help="feature catalog JSON (see schema in --help / module docstring)")
    src.add_argument("--demo", action="store_true", help="generate a synthetic card-fraud demo catalog")
    ap.add_argument("-o", "--output", default="feature_explorer.html", help="output HTML path")
    ap.add_argument("--title", help="override page title")
    ap.add_argument("--dump-json", metavar="PATH", help="also write the (derived) data JSON here")
    ap.add_argument("--seed", type=int, default=7, help="demo RNG seed")
    ap.add_argument("--rules", metavar="PATH", help="alert-rule overrides JSON (merged over the defaults and model.alert_rules)")
    args = ap.parse_args(argv)

    if args.demo:
        data = build_demo(args.seed)
    else:
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.title:
        data["title"] = args.title
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8")) if args.rules else None
    data = derive(data, rules)

    if args.dump_json:
        Path(args.dump_json).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.dump_json}")
    out = Path(args.output)
    out.write_text(render_html(data), encoding="utf-8")
    print(f"wrote {out}  ({len(data['features'])} features, {out.stat().st_size / 1024:.0f} KB)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --surface-2: #f3f2ee; --surface-3: #ebeae4;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --border-strong: rgba(11,11,11,0.18);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --s1-wash: rgba(42,120,214,0.10); --s1-band: rgba(42,120,214,0.16); --s1-band2: rgba(42,120,214,0.30);
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
  --good-text: #006300; --critical-text: #b42d2d; --warning-text: #8a5c00;
  --accent: #2a78d6; --focus: #2a78d6;
  --code-bg: #f3f2ee; --tk-kw: #6250d6; --tk-str: #1f7a4d; --tk-num: #b2541b; --tk-com: #898781; --tk-fn: #1c5cab;
  --shadow: 0 1px 2px rgba(0,0,0,.04), 0 4px 16px rgba(0,0,0,.06);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --surface-2: #222220; --surface-3: #2c2c2a;
    --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --border-strong: rgba(255,255,255,0.18);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --s1-wash: rgba(57,135,229,0.14); --s1-band: rgba(57,135,229,0.20); --s1-band2: rgba(57,135,229,0.38);
    --good-text: #0ca30c; --critical-text: #e66767; --warning-text: #fab219;
    --accent: #3987e5; --focus: #5598e7;
    --code-bg: #141413; --tk-kw: #9085e9; --tk-str: #4cc38a; --tk-num: #ec835a; --tk-com: #898781; --tk-fn: #6da7ec;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 4px 16px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --surface-2: #222220; --surface-3: #2c2c2a;
  --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --border-strong: rgba(255,255,255,0.18);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --s1-wash: rgba(57,135,229,0.14); --s1-band: rgba(57,135,229,0.20); --s1-band2: rgba(57,135,229,0.38);
  --good-text: #0ca30c; --critical-text: #e66767; --warning-text: #fab219;
  --accent: #3987e5; --focus: #5598e7;
  --code-bg: #141413; --tk-kw: #9085e9; --tk-str: #4cc38a; --tk-num: #ec835a; --tk-com: #898781; --tk-fn: #6da7ec;
  --shadow: 0 1px 2px rgba(0,0,0,.3), 0 4px 16px rgba(0,0,0,.35);
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body {
  background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased;
}
button, input, select { font: inherit; color: inherit; }
.mono, code, pre { font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, Consolas, monospace; }
.num { font-variant-numeric: tabular-nums; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; border-radius: 4px; }

/* ---------- top bar ---------- */
.topbar {
  position: sticky; top: 0; z-index: 30; display: flex; align-items: center; gap: 12px;
  height: 52px; padding: 0 16px; background: var(--surface); border-bottom: 1px solid var(--border);
}
.topbar h1 { font-size: 15px; font-weight: 650; margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.topbar .model { color: var(--muted); font-size: 12.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.topbar .spacer { flex: 1; }
.seg { display: inline-flex; background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: 2px; }
.seg button {
  border: 0; background: transparent; padding: 4px 10px; border-radius: 6px; cursor: pointer; color: var(--ink-2); font-size: 13px;
  white-space: nowrap;
}
.seg button[aria-pressed="true"] { background: var(--surface); color: var(--ink); box-shadow: 0 1px 2px rgba(0,0,0,.08); font-weight: 600; }
.seg button .count { color: var(--muted); font-weight: 400; margin-left: 4px; }
.btn {
  border: 1px solid var(--border-strong); background: var(--surface); border-radius: 8px; padding: 5px 10px; cursor: pointer;
  display: inline-flex; align-items: center; gap: 6px; font-size: 13px; white-space: nowrap;
}
.btn:hover { background: var(--surface-2); }
.btn.ghost { border-color: transparent; background: transparent; }
.btn.ghost:hover { background: var(--surface-2); }
.btn.small { padding: 2px 8px; font-size: 12px; }
.icon { width: 16px; height: 16px; flex: none; stroke: currentColor; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }

/* ---------- layout ---------- */
.layout { display: grid; grid-template-columns: 320px 1fr; height: calc(100vh - 52px); }
.sidebar { border-right: 1px solid var(--border); background: var(--surface); display: flex; flex-direction: column; min-height: 0; }
.main { overflow: auto; min-width: 0; }
.layout.no-sidebar { grid-template-columns: 0 1fr; }
.layout.no-sidebar .sidebar { display: none; }

.filters { padding: 12px; border-bottom: 1px solid var(--border); display: grid; gap: 8px; }
.search { position: relative; }
.search input {
  width: 100%; padding: 7px 30px 7px 30px; border: 1px solid var(--border-strong); border-radius: 8px; background: var(--page);
}
.search .icon { position: absolute; left: 9px; top: 9px; color: var(--muted); }
.search kbd { position: absolute; right: 8px; top: 7px; }
kbd {
  font: 11px ui-monospace, monospace; color: var(--muted); border: 1px solid var(--border-strong); border-bottom-width: 2px;
  border-radius: 4px; padding: 0 4px; background: var(--surface);
}
.row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.row select { flex: 1; min-width: 0; padding: 4px 6px; border: 1px solid var(--border-strong); border-radius: 6px; background: var(--surface); font-size: 12.5px; }
.chips { display: flex; flex-wrap: wrap; gap: 4px; }
.chip {
  border: 1px solid var(--border-strong); background: transparent; border-radius: 999px; padding: 1px 9px; font-size: 12px; cursor: pointer;
  color: var(--ink-2);
}
.chip[aria-pressed="true"] { background: var(--ink); color: var(--surface); border-color: var(--ink); }
.toggle { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--ink-2); cursor: pointer; user-select: none; }
.list-meta { padding: 6px 12px; font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; border-bottom: 1px solid var(--border); }
.flist { overflow: auto; flex: 1; padding: 4px 0; }
.fitem {
  display: grid; grid-template-columns: 1fr auto; gap: 2px 8px; padding: 7px 12px 7px 14px; cursor: pointer; border-left: 3px solid transparent;
}
.fitem:hover { background: var(--surface-2); }
.fitem[aria-selected="true"] { background: var(--s1-wash); border-left-color: var(--accent); }
.fitem .fname { font: 12.5px ui-monospace, monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fitem .fsub { font-size: 11.5px; color: var(--muted); grid-column: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.fitem .fright { grid-row: 1 / span 2; grid-column: 2; display: flex; align-items: center; gap: 6px; }
.pin { border: 0; background: transparent; padding: 2px; cursor: pointer; color: var(--muted); border-radius: 4px; display: inline-flex; }
.pin:hover { color: var(--ink); background: var(--surface-3); }
.pin[aria-pressed="true"] { color: var(--accent); }
.pin[aria-pressed="true"] .icon { fill: currentColor; }

/* ---------- badges ---------- */
.badge { display: inline-flex; align-items: center; gap: 4px; font-size: 11.5px; padding: 1px 7px; border-radius: 999px; border: 1px solid var(--border-strong); color: var(--ink-2); white-space: nowrap; }
.badge .dot { width: 7px; height: 7px; border-radius: 50%; }
.st-production .dot { background: var(--good); }
.st-candidate .dot { background: var(--s1); }
.st-deprecated .dot { background: var(--muted); }
.alert-ic { display: inline-flex; align-items: center; gap: 3px; font-size: 11.5px; font-weight: 600; color: var(--ink-2); }
.alert-ic svg { width: 14px; height: 14px; }
.lvl-critical { color: var(--critical); }
.lvl-warning { color: var(--warning); }
.lvl-info { color: var(--muted); }
.alert-ic .t { color: var(--ink-2); }

/* ---------- detail ---------- */
.detail { max-width: 1240px; margin: 0 auto; padding: 20px 24px 64px; }
.dhead { display: flex; gap: 16px; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; margin-bottom: 4px; }
.dhead h2 { font: 600 20px ui-monospace, monospace; margin: 0; word-break: break-all; }
.dhead .meta { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; align-items: center; }
.dhead .actions { display: flex; gap: 6px; flex-wrap: wrap; }
.desc { font-size: 15px; color: var(--ink); margin: 12px 0 4px; max-width: 80ch; }
.rationale { color: var(--ink-2); margin: 0 0 16px; max-width: 80ch; }
.tags { display: flex; gap: 4px; flex-wrap: wrap; }
.tag { font-size: 11.5px; color: var(--ink-2); background: var(--surface-3); padding: 1px 7px; border-radius: 4px; cursor: pointer; border: 0; }
.tag:hover { color: var(--ink); }

.alerts { display: grid; gap: 6px; margin: 0 0 16px; }
.alert { display: flex; gap: 8px; align-items: flex-start; padding: 8px 12px; border-radius: 8px; background: var(--surface); border: 1px solid var(--border); border-left-width: 4px; }
.alert.critical { border-left-color: var(--critical); }
.alert.warning { border-left-color: var(--warning); }
.alert.info { border-left-color: var(--muted); }
.alert svg { width: 16px; height: 16px; flex: none; margin-top: 2px; }
.alert b { font-weight: 600; margin-right: 4px; }
.alert .go { margin-left: auto; font-size: 12px; color: var(--accent); cursor: pointer; background: none; border: 0; padding: 0; white-space: nowrap; }

.tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
.tile .k { font-size: 12px; color: var(--ink-2); display: flex; justify-content: space-between; gap: 4px; }
.tile .v { font-size: 22px; font-weight: 600; margin-top: 2px; letter-spacing: -0.01em; }
.tile .d { font-size: 12px; color: var(--muted); margin-top: 2px; }
.tile .d.up-good, .tile .d.down-good { color: var(--good-text); }
.tile .d.up-bad, .tile .d.down-bad { color: var(--critical-text); }
.tile svg.spark { display: block; margin-top: 6px; width: 100%; height: 26px; }

.section-bar { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin: 4px 0 12px; position: sticky; top: 0; background: var(--page); padding: 8px 0; z-index: 5; border-bottom: 1px solid var(--border); }
.section-bar .jump { border: 0; background: none; color: var(--ink-2); font-size: 12.5px; padding: 3px 8px; border-radius: 6px; cursor: pointer; }
.section-bar .jump:hover { background: var(--surface-3); color: var(--ink); }
.section-bar .spacer { flex: 1; }

.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; margin-bottom: 14px; box-shadow: var(--shadow); scroll-margin-top: 104px; }
.card > header { display: flex; align-items: center; gap: 8px; padding: 10px 14px; cursor: pointer; user-select: none; }
.card > header h3 { margin: 0; font-size: 14px; font-weight: 650; }
.card > header .sub { color: var(--muted); font-size: 12.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.card > header .tools { margin-left: auto; display: flex; gap: 4px; align-items: center; }
.card > header .chev { transition: transform .15s; color: var(--muted); }
.card.collapsed > header .chev { transform: rotate(-90deg); }
.card.collapsed > .body { display: none; }
.card > .body { padding: 2px 14px 14px; }
.card .help { color: var(--ink-2); font-size: 12.5px; margin: 0 0 10px; max-width: 90ch; }
.empty { color: var(--muted); font-style: italic; padding: 12px 0; }

.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--ink-2); margin: 0 0 6px; }
.legend .sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -2px; margin-right: 5px; }
.legend .ln { display: inline-block; width: 16px; height: 0; border-top: 2px solid; vertical-align: 3px; margin-right: 5px; }
.legend .ln.dash { border-top-style: dashed; }
.chart { width: 100%; position: relative; }
.chart svg { display: block; width: 100%; overflow: visible; }
.chart text { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.chart .ref-label { fill: var(--ink-2); font-size: 11px; }
.ctl-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.subhead { font-size: 12.5px; font-weight: 600; color: var(--ink-2); margin: 0 0 6px; }

/* code */
.codewrap { position: relative; }
.codewrap pre {
  margin: 0; background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; overflow: auto;
  font-size: 12.5px; line-height: 1.55; max-height: 520px; tab-size: 4;
}
.codewrap .copy { position: absolute; top: 8px; right: 8px; }
.tk-kw { color: var(--tk-kw); font-weight: 600; }
.tk-str { color: var(--tk-str); }
.tk-num { color: var(--tk-num); }
.tk-com { color: var(--tk-com); font-style: italic; }
.tk-fn { color: var(--tk-fn); }
.lnum { color: var(--muted); user-select: none; display: inline-block; width: 2.4em; text-align: right; margin-right: 12px; opacity: .6; }

dl.kv { display: grid; grid-template-columns: max-content 1fr; gap: 6px 16px; margin: 0; font-size: 13px; }
dl.kv dt { color: var(--ink-2); }
dl.kv dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
.link { color: var(--accent); cursor: pointer; background: none; border: 0; padding: 0; font: 12.5px ui-monospace, monospace; text-align: left; }
.link:hover { text-decoration: underline; }

.corr { display: grid; grid-template-columns: minmax(0, 1fr) 160px 52px; gap: 6px 10px; align-items: center; font-size: 12.5px; }
.corr .bar { position: relative; height: 12px; }
.corr .bar::before { content: ""; position: absolute; left: 50%; top: -2px; bottom: -2px; border-left: 1px solid var(--axis); }
.corr .bar i { position: absolute; top: 0; height: 12px; background: var(--s1); border-radius: 3px; }
.corr .bar i.neg { background: var(--s2); }

.timeline-list { list-style: none; padding: 0; margin: 0; border-left: 2px solid var(--grid); }
.timeline-list li { position: relative; padding: 0 0 12px 16px; }
.timeline-list li::before { content: ""; position: absolute; left: -6px; top: 5px; width: 10px; height: 10px; border-radius: 50%; background: var(--surface); border: 2px solid var(--s1); }
.timeline-list .when { font-size: 12px; color: var(--muted); }

table.data { border-collapse: collapse; width: 100%; font-size: 12.5px; }
table.data th, table.data td { padding: 5px 8px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
table.data th:first-child, table.data td:first-child { text-align: left; }
table.data th { color: var(--ink-2); font-weight: 600; position: sticky; top: 0; background: var(--surface); }
.tablewrap { overflow: auto; max-height: 420px; border: 1px solid var(--border); border-radius: 8px; }

/* catalog */
.catalog { padding: 16px 20px 48px; }
.catalog .summary-tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; margin-bottom: 14px; }
.ctable-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: auto; box-shadow: var(--shadow); }
table.ctable { border-collapse: collapse; width: 100%; font-size: 12.5px; }
table.ctable th, table.ctable td { padding: 6px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; text-align: right; }
table.ctable th { position: sticky; top: 0; background: var(--surface); color: var(--ink-2); font-weight: 600; cursor: pointer; user-select: none; z-index: 1; }
table.ctable th.l, table.ctable td.l { text-align: left; }
table.ctable th .arrow { color: var(--accent); }
table.ctable tbody tr { cursor: pointer; }
table.ctable tbody tr:hover { background: var(--surface-2); }
table.ctable td .fname { font-family: ui-monospace, monospace; }
table.ctable svg.spark { width: 72px; height: 20px; vertical-align: middle; margin-left: 6px; }
.shapbar { display: inline-block; height: 8px; background: var(--s1); border-radius: 0 3px 3px 0; vertical-align: middle; margin-left: 6px; }
.cell-bad { color: var(--critical-text); font-weight: 600; }
.cell-warn { color: var(--warning-text); font-weight: 600; }

/* popover */
.pop { position: relative; }
.pop-panel {
  position: absolute; right: 0; top: calc(100% + 6px); background: var(--surface); border: 1px solid var(--border-strong); border-radius: 10px;
  box-shadow: var(--shadow); padding: 8px; min-width: 240px; z-index: 50; display: none;
}
.pop.open .pop-panel { display: block; }
.pop-panel label { display: flex; gap: 8px; align-items: center; padding: 4px 6px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.pop-panel label:hover { background: var(--surface-2); }
.pop-panel .foot { border-top: 1px solid var(--border); margin-top: 6px; padding-top: 6px; display: flex; gap: 6px; justify-content: space-between; }
.pop-panel h4 { margin: 4px 6px 6px; font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }

/* tooltip */
#tip {
  position: fixed; pointer-events: none; z-index: 100; background: var(--surface); color: var(--ink); border: 1px solid var(--border-strong);
  border-radius: 8px; box-shadow: var(--shadow); padding: 7px 10px; font-size: 12px; display: none; min-width: 120px; max-width: 320px;
}
#tip .tt { font-weight: 600; margin-bottom: 3px; }
#tip .tr { display: flex; gap: 10px; justify-content: space-between; align-items: center; }
#tip .tr .lbl { color: var(--ink-2); display: inline-flex; align-items: center; gap: 5px; }
#tip .tr .lbl i { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
#tip .tr .val { font-variant-numeric: tabular-nums; font-weight: 600; }
#toast { position: fixed; bottom: 18px; left: 50%; transform: translateX(-50%); background: var(--ink); color: var(--surface); padding: 6px 14px; border-radius: 8px; font-size: 13px; opacity: 0; transition: opacity .2s; pointer-events: none; z-index: 120; }
#toast.show { opacity: 1; }

.compare-grid { display: grid; gap: 14px; }
.hint { color: var(--muted); font-size: 12px; }
.kbd-help { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; font-size: 12.5px; padding: 4px 6px; }

@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; height: auto; }
  .sidebar { max-height: 45vh; border-right: 0; border-bottom: 1px solid var(--border); }
  .grid2 { grid-template-columns: 1fr; }
  .topbar .model, #sectionsBtn span, .topbar h1, #helpPop { display: none; }
  .topbar .spacer { display: none; }
  .topbar { gap: 6px; padding: 0 8px; }
  .seg button { padding: 4px 7px; }
  .pop-panel { position: fixed; left: 16px; right: 16px; top: 58px; min-width: 0; }
  .detail { padding: 14px 16px 48px; }
  .catalog { padding: 12px 16px 40px; }
}
@media print {
  .topbar, .sidebar, .section-bar, .card > header .tools { display: none !important; }
  .layout { display: block; height: auto; }
  .card { break-inside: avoid; box-shadow: none; }
}

/* ================= v2 additions ================= */
:root { --seq-lo: #f0efec; --seq-hi: #1c5cab; --div-neg: #c93a3a; --div-mid: #f0efec; --div-pos: #1c5cab; --hatch: #c3c2b7; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --seq-lo: #2c2c2a; --seq-hi: #6da7ec; --div-neg: #e66767; --div-mid: #383835; --div-pos: #6da7ec; --hatch: #4a4a46; }
}
:root[data-theme="dark"] { --seq-lo: #2c2c2a; --seq-hi: #6da7ec; --div-neg: #e66767; --div-mid: #383835; --div-pos: #6da7ec; --hatch: #4a4a46; }

#tip .tn { color: var(--ink-2); margin-top: 4px; padding-top: 4px; border-top: 1px solid var(--border); max-width: 300px; white-space: normal; }
#tip .tn b { color: var(--ink); }
.chart .ev-label { fill: var(--ink-2); font-size: 10px; }

.seg-select { padding: 3px 6px; border: 1px solid var(--border-strong); border-radius: 6px; background: var(--surface); font-size: 12.5px; max-width: 220px; }
.ctl-label { font-size: 12.5px; color: var(--ink-2); display: inline-flex; gap: 6px; align-items: center; }
.seg-banner { display: flex; gap: 8px; align-items: center; padding: 6px 10px; margin: 0 0 10px; border-radius: 8px; background: var(--s1-wash); font-size: 12.5px; }
.seg-banner b { font-family: ui-monospace, monospace; }

.grad { display: inline-block; width: 120px; height: 10px; border-radius: 3px; vertical-align: middle; margin: 0 6px; }
.grad-legend { font-size: 11.5px; color: var(--muted); display: inline-flex; align-items: center; }
.hm text { fill: var(--ink-2); font-size: 11px; font-variant-numeric: tabular-nums; }
.hm .rowlab { cursor: pointer; }
.hm .rowlab:hover { fill: var(--accent); }
.hm .rowlab.sel { fill: var(--accent); font-weight: 600; }

/* lineage */
.lg-wrap { min-width: 0; overflow: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--surface); }
.lg-wrap svg { display: block; }
.lg-node { cursor: pointer; }
.lg-node rect { stroke-width: 1.2; }
.lg-node text { font: 11.5px ui-monospace, monospace; fill: var(--ink); }
.lg-node text.r { fill: var(--muted); font-family: system-ui, sans-serif; font-size: 10.5px; }
.lg-edge { fill: none; stroke: var(--axis); stroke-width: 1.3; }
svg.hl .lg-node:not(.on) { opacity: .18; }
svg.hl .lg-edge:not(.on) { opacity: .08; }
svg.hl .lg-edge.on { stroke: var(--accent); stroke-width: 1.8; }
.lg-node.match rect { stroke: var(--accent); stroke-width: 2.4; }
.lg-node.focus rect { stroke: var(--accent); stroke-width: 2.4; }
.lg-node.dep { opacity: .55; }
.lg-layout { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 14px; align-items: start; }
.side-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; position: sticky; top: 12px; max-height: calc(100vh - 100px); overflow: auto; }
.side-panel h4 { margin: 0 0 4px; font: 600 13px ui-monospace, monospace; word-break: break-all; }
.side-panel ul { list-style: none; padding: 0; margin: 6px 0 0; }
.side-panel li { display: flex; justify-content: space-between; gap: 8px; padding: 3px 0; border-bottom: 1px solid var(--grid); font-size: 12.5px; }

/* redundancy */
.red-layout { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 14px; align-items: start; }
.rgroup { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; }
.rgroup header { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 12.5px; color: var(--ink-2); }
.rgroup .mem { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; gap: 8px; align-items: center; padding: 3px 0; font-size: 12.5px; border-top: 1px solid var(--grid); }
.keep { font-size: 11px; font-weight: 600; padding: 0 6px; border-radius: 4px; background: var(--s1-wash); color: var(--accent); }
.review { font-size: 11px; padding: 0 6px; border-radius: 4px; background: var(--surface-3); color: var(--ink-2); }
input[type=range] { accent-color: var(--accent); vertical-align: middle; }

/* rules */
.rules-tbl { border-collapse: collapse; font-size: 12.5px; width: 100%; }
.rules-tbl th, .rules-tbl td { padding: 4px 6px; border-bottom: 1px solid var(--grid); text-align: left; white-space: nowrap; }
.rules-tbl th { color: var(--ink-2); font-weight: 600; }
.rules-tbl input { width: 64px; padding: 2px 4px; border: 1px solid var(--border-strong); border-radius: 4px; background: var(--page); font-variant-numeric: tabular-nums; }
.rules-tbl input.edited { border-color: var(--accent); background: var(--s1-wash); }
#rulesPanel { min-width: 420px; max-height: 70vh; overflow: auto; }
.src { font-size: 11px; padding: 0 6px; border-radius: 4px; background: var(--surface-3); color: var(--ink-2); }
.src.feature { background: var(--s1-wash); color: var(--accent); }
.src.edited { background: rgba(250,178,25,.18); color: var(--warning-text); }
.ok-text { color: var(--good-text); }

@media (max-width: 1100px) {
  .lg-layout, .red-layout { grid-template-columns: 1fr; }
  .side-panel { position: static; max-height: none; }
}
@media (max-width: 900px) {
  .topbar .seg { overflow-x: auto; max-width: 52vw; }
  #rulesPanel { min-width: 0; }
  #rulesBtn span { display: none; }
}
</style>
</head>
<body>
<div class="topbar">
  <button class="btn ghost" id="toggleSidebar" title="Toggle feature list ( [ )" aria-label="Toggle feature list">
    <svg class="icon" viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/></svg>
  </button>
  <div style="min-width:0">
    <h1 id="title"></h1>
    <div class="model" id="modelInfo"></div>
  </div>
  <div class="spacer"></div>
  <div class="seg" role="group" aria-label="View">
    <button data-view="catalog" title="Catalog (c)">Catalog</button>
    <button data-view="detail" title="Feature detail (d)">Feature</button>
    <button data-view="compare" title="Compare pinned (p)">Compare<span class="count" id="pinCount"></span></button>
    <button data-view="lineage" title="Lineage graph (l)">Lineage</button>
    <button data-view="redundancy" title="Correlation &amp; redundancy (r)">Redundancy</button>
  </div>
  <div class="pop" id="rulesPop">
    <button class="btn" id="rulesBtn" title="Alert rules">
      <svg class="icon" viewBox="0 0 24 24"><path d="M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9M10.3 21a1.9 1.9 0 0 0 3.4 0"/></svg><span>Rules</span>
    </button>
    <div class="pop-panel" id="rulesPanel"></div>
  </div>
  <div class="pop" id="sectionsPop">
    <button class="btn" id="sectionsBtn" title="Choose which sections are shown">
      <svg class="icon" viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h10"/></svg><span>Sections</span>
    </button>
    <div class="pop-panel" id="sectionsPanel"></div>
  </div>
  <div class="pop" id="helpPop">
    <button class="btn ghost" id="helpBtn" title="Keyboard shortcuts (?)" aria-label="Keyboard shortcuts">
      <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .8-1 1.5v.7M12 17h.01"/></svg>
    </button>
    <div class="pop-panel">
      <h4>Keyboard</h4>
      <div class="kbd-help">
        <kbd>/</kbd><span>Search features</span>
        <kbd>j</kbd><span>Next feature</span>
        <kbd>k</kbd><span>Previous feature</span>
        <kbd>c</kbd><span>Catalog view</span>
        <kbd>d</kbd><span>Feature view</span>
        <kbd>p</kbd><span>Compare pinned</span>
        <kbd>l</kbd><span>Lineage graph</span>
        <kbd>r</kbd><span>Redundancy</span>
        <kbd>x</kbd><span>Pin / unpin current</span>
        <kbd>e</kbd><span>Expand / collapse all sections</span>
        <kbd>[</kbd><span>Toggle feature list</span>
        <kbd>t</kbd><span>Toggle theme</span>
        <kbd>Esc</kbd><span>Clear search / close menus</span>
      </div>
    </div>
  </div>
  <button class="btn ghost" id="themeBtn" title="Toggle theme (t)" aria-label="Toggle theme">
    <svg class="icon" viewBox="0 0 24 24"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
  </button>
</div>

<div class="layout" id="layout">
  <aside class="sidebar">
    <div class="filters">
      <div class="search">
        <svg class="icon" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
        <input id="q" type="search" placeholder="Search features, code, tags…" autocomplete="off" aria-label="Search features">
        <kbd>/</kbd>
      </div>
      <div class="chips" id="groupChips" aria-label="Filter by group"></div>
      <div class="row">
        <select id="fStatus" aria-label="Status"><option value="">All statuses</option></select>
        <select id="fType" aria-label="Data type"><option value="">All types</option></select>
      </div>
      <div class="row">
        <select id="sortBy" aria-label="Sort">
          <option value="rank">Sort: importance rank</option>
          <option value="name">Sort: name</option>
          <option value="auc">Sort: AUC (high→low)</option>
          <option value="recall_at_1pct_fpr">Sort: recall @ FPR (high→low)</option>
          <option value="usd_at_risk">Sort: $ at risk if dropped</option>
          <option value="psi">Sort: PSI (high→low)</option>
          <option value="missing_rate">Sort: missing rate (high→low)</option>
          <option value="alerts">Sort: alert severity</option>
          <option value="group">Sort: group</option>
        </select>
      </div>
      <div class="row" style="justify-content:space-between">
        <label class="toggle"><input type="checkbox" id="fAlerts"> Only with alerts</label>
        <button class="btn ghost small" id="clearFilters">Clear</button>
      </div>
    </div>
    <div class="list-meta"><span id="listCount"></span><span class="hint">j / k to move</span></div>
    <div class="flist" id="flist" role="listbox" aria-label="Features"></div>
  </aside>
  <main class="main" id="main"></main>
</div>
<div id="tip" role="tooltip"></div>
<div id="toast" role="status" aria-live="polite"></div>

<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
"use strict";
const DATA = JSON.parse(document.getElementById("data").textContent);
const FEATS = DATA.features || [];
const BYNAME = Object.fromEntries(FEATS.map(f => [f.name, f]));
const MODEL = DATA.model || {};
const NBASE = MODEL.baseline_periods || 3;
const MAT = MODEL.label_maturity_periods || 0;
const FPR = MODEL.fpr_budget || 0.01;
const FPR_TXT = +(FPR * 100).toFixed(2) + "%";

/* ---------------------------------------------------------------- utils */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const isNum = v => typeof v === "number" && isFinite(v);
const mean = a => { const b = (a || []).filter(isNum); return b.length ? b.reduce((x, y) => x + y, 0) / b.length : null; };
const clone = o => JSON.parse(JSON.stringify(o));
function deepMerge(a, b) {
  const out = Object.assign({}, a);
  Object.entries(b || {}).forEach(([k, v]) => { out[k] = v && typeof v === "object" && !Array.isArray(v) && out[k] && typeof out[k] === "object" ? deepMerge(out[k], v) : v; });
  return out;
}
const store = {
  get(k, d) { try { const v = localStorage.getItem("fx2:" + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("fx2:" + k, JSON.stringify(v)); } catch (e) {} }
};
const fmt = {
  auc: v => isNum(v) ? v.toFixed(3) : "—",
  pct: v => isNum(v) ? (Math.abs(v) < 0.001 && v !== 0 ? (v * 100).toFixed(2) : (v * 100).toFixed(1)) + "%" : "—",
  rate: v => isNum(v) ? (v < 0.01 ? (v * 100).toFixed(2) : (v * 100).toFixed(1)) + "%" : "—",
  int: v => isNum(v) ? Math.round(v).toLocaleString() : "—",
  usd: v => !isNum(v) ? "—" : Math.abs(v) >= 1e6 ? "$" + (v / 1e6).toFixed(2) + "M" : Math.abs(v) >= 1e3 ? "$" + (v / 1e3).toFixed(0) + "k" : "$" + v.toFixed(0),
  num: v => {
    if (!isNum(v)) return "—";
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (a >= 1e4) return (v / 1e3).toFixed(0) + "k";
    if (a >= 1000) return (v / 1e3).toFixed(1) + "k";
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return "0";
    return v.toPrecision(2);
  }
};
const METRIC = {
  auc: { label: "AUC", fmt: fmt.auc, better: "up", labels: true, help: "Univariate ROC-AUC of the feature alone vs. the fraud label (oriented so ≥ 0.5)." },
  ks: { label: "KS", fmt: fmt.auc, better: "up", labels: true, help: "Kolmogorov–Smirnov: max gap between fraud and legit cumulative distributions." },
  iv: { label: "IV", fmt: fmt.auc, better: "up", labels: true, help: "Information Value over the baseline bins (< 0.02 useless, 0.1–0.3 medium, > 0.3 strong)." },
  recall_at_1pct_fpr: { label: `Recall @${FPR_TXT} FPR`, short: "Recall", fmt: fmt.pct, better: "up", labels: true, pp: true,
    help: `Share of fraud transactions caught when the feature alone flags the riskiest transactions, capped so at most ${FPR_TXT} of legit traffic is flagged.` },
  value_recall_at_1pct_fpr: { label: `$ recall @${FPR_TXT} FPR`, short: "$ recall", fmt: fmt.pct, better: "up", labels: true, pp: true,
    help: `Share of fraud dollars caught at the same ${FPR_TXT} false-positive budget — what the fraud-loss P&L cares about.` },
  psi: { label: "PSI", fmt: fmt.auc, better: "down", help: "Population Stability Index vs. the baseline window (< 0.1 stable, 0.1–0.25 moderate, > 0.25 major)." },
  missing_rate: { label: "Missing", fmt: fmt.pct, better: "down", pp: true, help: "Share of transactions where the feature is null." },
  share: { label: "Volume share", fmt: fmt.pct, help: "Share of transactions falling in this segment." }
};
const LABEL_METRICS = Object.keys(METRIC).filter(k => METRIC[k].labels);
const SEV = { critical: 3, warning: 2, info: 1 };
function maxSev(f) { return Math.max(0, ...(f.alerts || []).map(a => SEV[a.level] || 0)); }
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function shortPeriod(p) { const m = /^(\d{4})-(\d{2})$/.exec(p); if (!m) return p; return ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][+m[2] - 1] + " ’" + m[1].slice(2); }
const PERIODS = ((FEATS.find(f => f.timeline && f.timeline.periods) || {}).timeline || {}).periods || [];
const MATURE_IDX = PERIODS.length - 1 - MAT;

/* colour scales for heatmaps (interpolated between theme tokens) */
function hexRgb(h) { h = h.replace("#", ""); if (h.length === 3) h = h.split("").map(c => c + c).join(""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); }
function mixRgb(a, b, t) { const A = hexRgb(a), B = hexRgb(b); t = Math.max(0, Math.min(1, t)); return A.map((x, i) => Math.round(x + (B[i] - x) * t)); }
const rgbStr = c => `rgb(${c.join(",")})`;
function inkOn(c) { const L = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255; return L > 0.55 ? "#0b0b0b" : "#f0efec"; }
function seqColor(t) { return mixRgb(cssVar("--seq-lo"), cssVar("--seq-hi"), t); }
function divColor(t) { return t < 0 ? mixRgb(cssVar("--div-mid"), cssVar("--div-neg"), -t) : mixRgb(cssVar("--div-mid"), cssVar("--div-pos"), t); }
function gradLegend(lo, hi, from, to, mid) {
  const bg = mid ? `linear-gradient(90deg, ${from}, ${mid}, ${to})` : `linear-gradient(90deg, ${from}, ${to})`;
  return `<span class="grad-legend">${esc(lo)}<span class="grad" style="background:${bg}"></span>${esc(hi)}</span>`;
}

const ICON = {
  critical: '<svg viewBox="0 0 24 24" class="lvl-critical" aria-hidden="true"><path fill="currentColor" d="M12 2 1 21h22L12 2zm1 15h-2v-2h2v2zm0-4h-2V9h2v4z"/></svg>',
  warning: '<svg viewBox="0 0 24 24" class="lvl-warning" aria-hidden="true"><circle cx="12" cy="12" r="10" fill="currentColor"/><path d="M12 7v6M12 16.5v.5" stroke="#0b0b0b" stroke-width="2.2" stroke-linecap="round"/></svg>',
  info: '<svg viewBox="0 0 24 24" class="lvl-info" aria-hidden="true"><circle cx="12" cy="12" r="10" fill="currentColor"/><path d="M12 11v6M12 7.5v.5" stroke="#fff" stroke-width="2.2" stroke-linecap="round"/></svg>',
  pin: '<svg class="icon" viewBox="0 0 24 24"><path d="M12 17v5M8 3h8l-1 6 3 4H6l3-4-1-6z"/></svg>',
  chev: '<svg class="icon chev" viewBox="0 0 24 24"><path d="m6 9 6 6 6-6"/></svg>',
  eyeOff: '<svg class="icon" viewBox="0 0 24 24"><path d="M3 3l18 18M10.6 10.6a2 2 0 0 0 2.8 2.8M9.9 5.1A9.8 9.8 0 0 1 12 5c5 0 9 5 9 7a11 11 0 0 1-2.4 3.4M6.1 6.1C4 7.5 3 10 3 12c0 2 4 7 9 7a9.6 9.6 0 0 0 4-.9"/></svg>',
  copy: '<svg class="icon" viewBox="0 0 24 24"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>',
  link: '<svg class="icon" viewBox="0 0 24 24"><path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/></svg>',
  prev: '<svg class="icon" viewBox="0 0 24 24"><path d="m15 18-6-6 6-6"/></svg>',
  next: '<svg class="icon" viewBox="0 0 24 24"><path d="m9 18 6-6-6-6"/></svg>',
  graph: '<svg class="icon" viewBox="0 0 24 24"><circle cx="5" cy="6" r="2"/><circle cx="5" cy="18" r="2"/><circle cx="19" cy="12" r="2"/><path d="M7 6c6 0 6 6 10 6M7 18c6 0 6-6 10-6"/></svg>'
};
function alertBadge(f) {
  const real = (f.alerts || []).filter(x => x.level !== "info");
  if (!real.length) return f.muted ? `<span class="hint" title="Alerts muted">muted</span>` : "";
  const lvl = real.some(x => x.level === "critical") ? "critical" : "warning";
  return `<span class="alert-ic" title="${esc(real.map(x => x.message).join("\n"))}">${ICON[lvl]}<span class="t">${real.length}</span></span>`;
}
function statusBadge(s) { return `<span class="badge st-${esc(s)}"><span class="dot"></span>${esc(s)}</span>`; }
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 1600); }
function copyText(s, msg) {
  const done = () => toast(msg || "Copied");
  const fallback = () => { const ta = document.createElement("textarea"); ta.value = s; document.body.appendChild(ta); ta.select(); try { document.execCommand("copy"); done(); } catch (e) {} ta.remove(); };
  if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(s).then(done, fallback); else fallback();
}
function signed(d, digits, pp) { if (!isNum(d)) return "—"; return (d >= 0 ? "+" : "−") + (pp ? Math.abs(d * 100).toFixed(1) + "pp" : Math.abs(d).toFixed(digits == null ? 3 : digits)); }

/* ---------------------------------------------------------------- tooltip */
const tip = $("#tip");
function showTip(e, html) {
  tip.innerHTML = html; tip.style.display = "block";
  const r = tip.getBoundingClientRect(); let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
  tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
}
function hideTip() { tip.style.display = "none"; }
function tipRows(title, rows) {
  return `<div class="tt">${esc(title)}</div>` + rows.map(r => r.note != null
    ? `<div class="tn">${r.html ? r.note : esc(r.note)}</div>`
    : `<div class="tr"><span class="lbl">${r.color ? `<i style="background:${r.color}"></i>` : ""}${esc(r.label)}</span><span class="val">${esc(r.value)}</span></div>`).join("");
}

/* ---------------------------------------------------------------- derived data: kpis, alerts, correlations, events */
const RULES_FILE = MODEL.alert_rules || {};
let RULES = deepMerge(RULES_FILE, store.get("rules", {}));
const MUTED = new Set(store.get("muted", []));
const RULE_META = {
  psi: { label: "PSI (latest period)", fields: ["warning", "critical"], metric: "psi" },
  missing_increase: { label: "Missing-rate rise vs baseline", fields: ["warning", "critical"], metric: "missing_rate", unit: "abs" },
  missing_spike: { label: "Missing spike within lookback", fields: ["warning", "lookback"], metric: "missing_rate", unit: "abs" },
  auc_drop: { label: "AUC drop vs baseline (mature labels)", fields: ["warning", "critical"], metric: "auc" },
  recall_drop: { label: `Recall @${FPR_TXT} drop (mature labels)`, fields: ["warning", "critical"], metric: "recall_at_1pct_fpr", unit: "abs" },
  segment_psi: { label: "Segment PSI (hidden drift)", fields: ["warning", "critical", "min_share"], metric: "psi" }
};
function rulesFor(f) {
  const o = f.alert_overrides || {};
  const objs = Object.fromEntries(Object.entries(o).filter(([k, v]) => RULE_META[k] && v && typeof v === "object"));
  return deepMerge(RULES, objs);
}
function lastIdx(arr, upto) { for (let i = Math.min(upto, (arr || []).length - 1); i >= 0; i--) if (isNum(arr[i])) return i; return -1; }
function computeKpis(f) {
  const tl = f.timeline || {}, n = (tl.periods || []).length, k = {};
  [...LABEL_METRICS, "psi", "missing_rate"].forEach(m => {
    const s = tl[m] || [], upto = LABEL_METRICS.includes(m) ? n - 1 - MAT : n - 1, i = lastIdx(s, upto);
    k[m] = i >= 0 ? s[i] : null; k[m + "_idx"] = i; k[m + "_base"] = mean(s.slice(0, NBASE));
    k[m + "_naive"] = s.length ? s[s.length - 1] : null;
  });
  const lb = (RULES.missing_spike && RULES.missing_spike.lookback) || 6;
  k.psi_max6 = Math.max(...(tl.psi || []).slice(-6).filter(isNum), -Infinity); if (!isFinite(k.psi_max6)) k.psi_max6 = null;
  k.missing_max = Math.max(...(tl.missing_rate || []).slice(-lb).filter(isNum), -Infinity); if (!isFinite(k.missing_max)) k.missing_max = null;
  const ab = (f.model_behavior || {}).ablation || {};
  k.usd_at_risk = ab.usd_at_risk_per_month; k.shap = (f.importance || {}).shap_mean_abs;
  // worst segment drift in the latest period
  k.seg_psi = null; k.seg_key = null;
  Object.entries(f.segments || {}).forEach(([dim, vals]) => Object.entries(vals).forEach(([val, s]) => {
    const share = (s.share || [])[(s.share || []).length - 1], p = (s.psi || [])[(s.psi || []).length - 1];
    if (isNum(p) && isNum(share) && share >= ((RULES.segment_psi || {}).min_share || 0) && (k.seg_psi == null || p > k.seg_psi)) { k.seg_psi = p; k.seg_key = dim + "=" + val; k.seg_share = share; }
  }));
  f.kpis = k;
}
function computeAlerts(f) {
  const r = rulesFor(f), k = f.kpis, o = f.alert_overrides || {}, out = [];
  const disabled = new Set(o.disabled || []), on = key => r[key] && !disabled.has(key);
  f.muted = !!o.muted || MUTED.has(f.name);
  const lvl = (v, rule) => isNum(rule.critical) && v >= rule.critical ? "critical" : isNum(rule.warning) && v >= rule.warning ? "warning" : null;
  const P = i => PERIODS[i] || "";
  if (on("psi") && isNum(k.psi)) { const l = lvl(k.psi, r.psi); if (l) out.push({ level: l, metric: "psi", sec: "stability", message: `${l === "critical" ? "Major" : "Moderate"} population shift: PSI ${k.psi.toFixed(3)} in ${P(PERIODS.length - 1)}` }); }
  let missFired = false;
  if (on("missing_increase") && isNum(k.missing_rate) && isNum(k.missing_rate_base)) {
    const d = k.missing_rate - k.missing_rate_base, l = lvl(d, r.missing_increase);
    if (l) { missFired = true; out.push({ level: l, metric: "missing_rate", sec: "missing", message: `Missing rate up ${(d * 100).toFixed(1)}pp vs baseline (${fmt.pct(k.missing_rate_base)} → ${fmt.pct(k.missing_rate)})` }); }
  }
  if (!missFired && on("missing_spike") && isNum(k.missing_max) && isNum(k.missing_rate_base) && k.missing_max - k.missing_rate_base >= r.missing_spike.warning)
    out.push({ level: "warning", metric: "missing_rate", sec: "missing", message: `Missingness spike in last ${r.missing_spike.lookback} periods (peak ${fmt.pct(k.missing_max)}, baseline ${fmt.pct(k.missing_rate_base)})` });
  if (on("auc_drop") && isNum(k.auc) && isNum(k.auc_base)) {
    const d = k.auc_base - k.auc, l = lvl(d, r.auc_drop);
    if (l) out.push({ level: l, metric: "auc", sec: "performance", message: `Univariate AUC down ${d.toFixed(3)} vs baseline (${k.auc_base.toFixed(3)} → ${k.auc.toFixed(3)}, ${P(k.auc_idx)})` });
  }
  if (on("recall_drop") && isNum(k.recall_at_1pct_fpr) && isNum(k.recall_at_1pct_fpr_base)) {
    const d = k.recall_at_1pct_fpr_base - k.recall_at_1pct_fpr, l = lvl(d, r.recall_drop);
    if (l) out.push({ level: l, metric: "recall_at_1pct_fpr", sec: "performance", message: `Recall @${FPR_TXT} FPR down ${(d * 100).toFixed(1)}pp vs baseline (${fmt.pct(k.recall_at_1pct_fpr_base)} → ${fmt.pct(k.recall_at_1pct_fpr)})` });
  }
  if (on("segment_psi") && isNum(k.seg_psi)) {
    const l = lvl(k.seg_psi, r.segment_psi), overall = isNum(k.psi) ? k.psi : 0;
    if (l && k.seg_psi > 2 * overall && overall < ((r.psi || {}).warning || Infinity))
      out.push({ level: l, metric: "segment", sec: "segments", seg: k.seg_key, message: `Hidden drift in ${k.seg_key.replace("=", " = ")} (${fmt.pct(k.seg_share)} of volume): PSI ${k.seg_psi.toFixed(3)} while overall is ${overall.toFixed(3)}` });
  }
  if (f.muted) { f.mutedAlerts = out.length; out.length = 0; }
  if (f.status === "deprecated") out.push({ level: "info", metric: "status", message: "Deprecated — still in the model, scheduled for removal" });
  if (o.muted && o.note) out.push({ level: "info", metric: "status", message: "Alerts muted: " + o.note });
  f.alerts = out;
}
function recomputeAll() { FEATS.forEach(f => { computeKpis(f); computeAlerts(f); }); }

const CM = DATA.correlations && DATA.correlations.features ? DATA.correlations : null;
const CIDX = CM ? Object.fromEntries(CM.features.map((n, i) => [n, i])) : {};
function corr(a, b) { if (!CM) return null; const i = CIDX[a], j = CIDX[b]; return i == null || j == null ? null : CM.matrix[i][j]; }
FEATS.forEach(f => {
  if (!f.correlations && CM && CIDX[f.name] != null)
    f.correlations = CM.features.filter(n => n !== f.name).map(n => ({ feature: n, r: corr(f.name, n) })).filter(c => isNum(c.r)).sort((a, b) => Math.abs(b.r) - Math.abs(a.r)).slice(0, 8);
});

function periodIndex(date) {
  if (!date || !PERIODS.length) return -1;
  const d = String(date), L = PERIODS[0].length, key = d.slice(0, L);
  if (key < PERIODS[0] || key > PERIODS[PERIODS.length - 1]) return -1;
  let idx = -1; PERIODS.forEach((p, i) => { if (p <= key) idx = i; }); return idx;
}
const GLOBAL_EVENTS = (DATA.annotations || []).map(a => ({ i: periodIndex(a.date), label: a.label || "Event", text: a.text || "", date: a.date, kind: "global" })).filter(e => e.i >= 0);
function eventsFor(f) {
  if (!S.showEvents) return [];
  const own = f ? (f.changelog || []).map(c => ({ i: periodIndex(c.date), label: "v" + (c.version || "?") + " · " + (f.name), text: c.note, date: c.date, kind: "feature" })).filter(e => e.i >= 0) : [];
  return GLOBAL_EVENTS.concat(own);
}

// in-model importance rank per period
const RANK_TL = PERIODS.map((_, i) => {
  const xs = FEATS.map(f => [f.name, ((f.model_behavior || {}).importance_timeline || [])[i]]).filter(x => isNum(x[1])).sort((a, b) => b[1] - a[1]);
  return Object.fromEntries(xs.map((x, r) => [x[0], r + 1]));
});

/* ---------------------------------------------------------------- charts (inline SVG) */
function niceStep(range, n) {
  const raw = range / Math.max(1, n), p = Math.pow(10, Math.floor(Math.log10(raw || 1))), m = raw / p;
  return (m >= 7.5 ? 10 : m >= 3.5 ? 5 : m >= 1.5 ? 2 : 1) * p;
}
function ticks(min, max, n) {
  if (min === max) max = min + 1;
  const st = niceStep(max - min, n), lo = Math.floor(min / st) * st, hi = Math.ceil(max / st) * st, out = [];
  for (let v = lo; v <= hi + st * 1e-6; v += st) out.push(+v.toFixed(10));
  return out;
}
function seriesColor(i) { return `var(--s${(i % 4) + 1})`; }
function legendHTML(items, style) {
  return `<div class="legend"${style ? ` style="${style}"` : ""}>` + items.map(it =>
    it.kind === "line" ? `<span><span class="ln${it.dash ? " dash" : ""}" style="border-color:${it.color}"></span>${esc(it.label)}</span>`
      : it.kind === "hatch" ? `<span><span class="sw" style="background:repeating-linear-gradient(45deg, var(--hatch) 0 2px, transparent 2px 5px);border:1px solid var(--hatch)"></span>${esc(it.label)}</span>`
      : it.kind === "event" ? `<span><svg width="10" height="10" style="vertical-align:-1px;margin-right:5px"><path d="M5 0 10 5 5 10 0 5Z" fill="var(--ink-2)"/></svg>${esc(it.label)}</span>`
      : `<span><span class="sw" style="background:${it.color}"></span>${esc(it.label)}</span>`).join("") + `</div>`;
}
function xLabelStep(n, w) { return Math.max(1, Math.ceil(n / Math.max(2, Math.floor(w / 64)))); }
let UID = 0;

/** Line chart. cfg: {x, series:[{name, values, color, dash, width}], bands:[{lo,hi,fill,name}], yFmt, yMin, yMax,
 *  refs:[{y,label}], shadeBaseline, immature, events:[{i,label,text}], height, zeroBased, tipExtra(i)} */
function lineChart(el, cfg) {
  const W = Math.max(280, el.clientWidth || 600), H = cfg.height || 220, n = cfg.x.length;
  const ev = (cfg.events || []).filter(e => e.i >= 0 && e.i < n);
  const M = { l: 52, r: 16, t: ev.length ? 20 : 10, b: 26 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const vals = [];
  cfg.series.forEach(s => s.values.forEach(v => isNum(v) && vals.push(v)));
  (cfg.bands || []).forEach(b => { b.lo.forEach(v => isNum(v) && vals.push(v)); b.hi.forEach(v => isNum(v) && vals.push(v)); });
  (cfg.refs || []).forEach(r => vals.push(r.y));
  if (!vals.length) { el.innerHTML = `<div class="empty">No data</div>`; return; }
  let mn = cfg.yMin != null ? cfg.yMin : Math.min(...vals), mx = cfg.yMax != null ? cfg.yMax : Math.max(...vals);
  if (cfg.zeroBased) mn = Math.min(0, mn);
  if (mn === mx) { mn -= Math.abs(mn) * 0.1 || 0.5; mx += Math.abs(mx) * 0.1 || 0.5; }
  const pad = (mx - mn) * 0.06; if (cfg.yMin == null && !cfg.zeroBased) mn -= pad; if (cfg.yMax == null) mx += pad;
  const tk = ticks(mn, mx, Math.max(3, Math.floor(ih / 44)));
  mn = Math.min(mn, tk[0]); mx = Math.max(mx, tk[tk.length - 1]);
  const step_ = n > 1 ? iw / (n - 1) : iw;
  const X = i => M.l + (n <= 1 ? iw / 2 : i / (n - 1) * iw), Y = v => M.t + ih - (v - mn) / (mx - mn) * ih;
  const yF = cfg.yFmt || fmt.num, pid = "h" + (++UID);
  let s = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${esc(cfg.aria || "line chart")}">` +
    `<defs><pattern id="${pid}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="var(--hatch)" stroke-width="2"/></pattern></defs>`;
  if (cfg.shadeBaseline && n > 1) {
    const x1 = X(Math.min(NBASE, n) - 1) + step_ / 2;
    s += `<rect x="${M.l}" y="${M.t}" width="${Math.max(0, x1 - M.l)}" height="${ih}" fill="var(--surface-3)" opacity=".55"/>`;
    s += `<text x="${M.l + 4}" y="${M.t + 12}" class="ref-label">baseline</text>`;
  }
  const immStart = cfg.immature && MAT > 0 && n > MAT ? n - MAT : null;
  if (immStart != null) {
    const x0 = X(immStart) - step_ / 2;
    s += `<rect x="${x0}" y="${M.t}" width="${M.l + iw - x0 + 4}" height="${ih}" fill="url(#${pid})" opacity=".7"/>`;
    s += `<text x="${M.l + iw}" y="${M.t + ih - 6}" text-anchor="end" class="ref-label">labels maturing</text>`;
  }
  tk.forEach(t => { s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)" stroke-width="1"/><text x="${M.l - 8}" y="${Y(t) + 3.5}" text-anchor="end">${esc(yF(t))}</text>`; });
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${M.t + ih}" y2="${M.t + ih}" stroke="var(--axis)"/>`;
  const step = xLabelStep(n, iw);
  cfg.x.forEach((p, i) => { if (i % step === 0) s += `<text x="${X(i)}" y="${H - 8}" text-anchor="middle">${esc(shortPeriod(p))}</text>`; });
  [...new Set(ev.map(e => e.i))].forEach(i => {
    s += `<line x1="${X(i)}" x2="${X(i)}" y1="${M.t - 6}" y2="${M.t + ih}" stroke="var(--muted)" stroke-dasharray="2 3"/>`;
    s += `<path d="M${X(i)},${M.t - 14}l5,5l-5,5l-5,-5z" fill="var(--ink-2)"/>`;
  });
  (cfg.bands || []).forEach(b => {
    let d = "", seg = [];
    const flush = () => { if (seg.length) { d += "M" + seg.map(i => `${X(i)},${Y(b.hi[i])}`).join("L") + "L" + seg.slice().reverse().map(i => `${X(i)},${Y(b.lo[i])}`).join("L") + "Z"; seg = []; } };
    for (let i = 0; i < n; i++) { if (isNum(b.lo[i]) && isNum(b.hi[i])) seg.push(i); else flush(); }
    flush();
    s += `<path d="${d}" fill="${b.fill}"/>`;
  });
  (cfg.refs || []).forEach(r => {
    s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(r.y)}" y2="${Y(r.y)}" stroke="var(--ink-2)" stroke-dasharray="4 3" stroke-width="1"/>`;
    s += `<text x="${W - M.r - 2}" y="${Y(r.y) - 4}" text-anchor="end" class="ref-label">${esc(r.label)}</text>`;
  });
  cfg.series.forEach(se => {
    let d = "", pen = false;
    se.values.forEach((v, i) => { if (isNum(v)) { d += (pen ? "L" : "M") + X(i) + "," + Y(v); pen = true; } else pen = false; });
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.width || 2}" stroke-linejoin="round" stroke-linecap="round"${se.dash ? ' stroke-dasharray="5 4"' : ""}/>`;
    se.values.forEach((v, i) => { if (isNum(v) && !isNum(se.values[i - 1]) && !isNum(se.values[i + 1])) s += `<circle cx="${X(i)}" cy="${Y(v)}" r="3" fill="${se.color}"/>`; });
  });
  s += `<g class="hover" style="display:none"><line class="xh" y1="${M.t}" y2="${M.t + ih}" stroke="var(--axis)" stroke-width="1"/>` +
    cfg.series.map((se, k) => `<circle class="hd" data-k="${k}" r="4.5" fill="${se.color}" stroke="var(--surface)" stroke-width="2"/>`).join("") + `</g>`;
  s += `<rect class="hit" x="${M.l - 6}" y="${M.t - 16}" width="${iw + 12}" height="${ih + 16}" fill="transparent"/></svg>`;
  el.innerHTML = s;
  const svg = el.firstChild, g = svg.querySelector(".hover"), hit = svg.querySelector(".hit");
  hit.addEventListener("mousemove", e => {
    const r = svg.getBoundingClientRect(), mx_ = (e.clientX - r.left) * (W / r.width);
    const i = Math.max(0, Math.min(n - 1, Math.round((mx_ - M.l) / (iw / Math.max(1, n - 1)))));
    g.style.display = ""; const xl = g.querySelector(".xh"); xl.setAttribute("x1", X(i)); xl.setAttribute("x2", X(i));
    $$(".hd", g).forEach(c => { const v = cfg.series[+c.dataset.k].values[i]; if (isNum(v)) { c.style.display = ""; c.setAttribute("cx", X(i)); c.setAttribute("cy", Y(v)); } else c.style.display = "none"; });
    const rows = cfg.series.map(se => ({ label: se.name, value: yF(se.values[i]), color: se.color }));
    (cfg.bands || []).forEach(b => b.name && rows.push({ label: b.name, value: `${yF(b.lo[i])} – ${yF(b.hi[i])}` }));
    if (cfg.tipExtra) rows.push(...cfg.tipExtra(i));
    if (immStart != null && i >= immStart) rows.push({ note: "Labels still maturing — recent fraud not yet reported, so label-based metrics read low." });
    ev.filter(x => x.i === i).forEach(x => rows.push({ note: `<b>◆ ${esc(x.label)}</b> ${esc(x.date || "")}<br>${esc(x.text)}`, html: true }));
    const tag = i < NBASE && cfg.shadeBaseline ? " · baseline" : "";
    showTip(e, tipRows(cfg.x[i] + tag, rows));
  });
  hit.addEventListener("mouseleave", () => { g.style.display = "none"; hideTip(); });
}

/** Grouped vertical bars. cfg: {cats, series:[{name, values, color}], yFmt, height, refs, rotate, catTitle, tipExtra(i)} */
function barChart(el, cfg) {
  const W = Math.max(280, el.clientWidth || 600), n = cfg.cats.length;
  const longest = Math.max(...cfg.cats.map(c => String(c).length));
  const slot0 = (W - 68) / Math.max(1, n);
  const rotate = cfg.rotate != null ? cfg.rotate : longest * 6.2 > slot0;
  const M = { l: 52, r: 12, t: 10, b: rotate ? Math.min(90, 14 + longest * 5.2) : 26 };
  const H = (cfg.height || 220) + (rotate ? M.b - 26 : 0);
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const vals = []; cfg.series.forEach(s => s.values.forEach(v => isNum(v) && vals.push(v)));
  (cfg.refs || []).forEach(r => vals.push(r.y));
  if (!vals.length) { el.innerHTML = `<div class="empty">No data</div>`; return; }
  const tk = ticks(0, Math.max(...vals) * 1.05 || 1, Math.max(3, Math.floor(ih / 44)));
  const mx = tk[tk.length - 1], Y = v => M.t + ih - v / mx * ih;
  const slot = iw / n, ns = cfg.series.length, gap = 2;
  const groupW = Math.min(slot * 0.78, 18 * ns + gap * (ns - 1) + 40), bw = Math.max(2, (groupW - gap * (ns - 1)) / ns);
  const yF = cfg.yFmt || fmt.num;
  let s = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${esc(cfg.aria || "bar chart")}">`;
  tk.forEach(t => { s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)"/><text x="${M.l - 8}" y="${Y(t) + 3.5}" text-anchor="end">${esc(yF(t))}</text>`; });
  cfg.cats.forEach((c, i) => {
    const gx = M.l + slot * i + (slot - (bw * ns + gap * (ns - 1))) / 2;
    cfg.series.forEach((se, k) => {
      const v = se.values[i]; if (!isNum(v) || v <= 0) return;
      const x = gx + k * (bw + gap), y = Y(v), h = M.t + ih - y, r = Math.min(4, bw / 2, h);
      s += `<path d="M${x},${M.t + ih}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${M.t + ih}Z" fill="${se.color}"/>`;
    });
    const cx = M.l + slot * i + slot / 2, lab = String(c);
    if (rotate) s += `<text transform="translate(${cx + 3},${M.t + ih + 10}) rotate(-40)" text-anchor="end">${esc(lab.length > 22 ? lab.slice(0, 21) + "…" : lab)}</text>`;
    else s += `<text x="${cx}" y="${H - 8}" text-anchor="middle">${esc(lab)}</text>`;
    s += `<rect class="hit" data-i="${i}" x="${M.l + slot * i}" y="${M.t}" width="${slot}" height="${ih}" fill="transparent"/>`;
  });
  (cfg.refs || []).forEach(r => {
    s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(r.y)}" y2="${Y(r.y)}" stroke="var(--ink-2)" stroke-dasharray="4 3"/>`;
    s += `<text x="${W - M.r - 2}" y="${Y(r.y) - 4}" text-anchor="end" class="ref-label">${esc(r.label)}</text>`;
  });
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${M.t + ih}" y2="${M.t + ih}" stroke="var(--axis)"/></svg>`;
  el.innerHTML = s;
  $$(".hit", el).forEach(h => {
    h.addEventListener("mousemove", e => {
      const i = +h.dataset.i; h.setAttribute("fill", "var(--s1-wash)");
      const rows = cfg.series.map(se => ({ label: se.name, value: yF(se.values[i]), color: se.color }));
      if (cfg.tipExtra) rows.push(...cfg.tipExtra(i));
      showTip(e, tipRows((cfg.catTitle ? cfg.catTitle + ": " : "") + cfg.cats[i], rows));
    });
    h.addEventListener("mouseleave", () => { h.setAttribute("fill", "transparent"); hideTip(); });
  });
}

/** 100% stacked bars over time. cfg: {x, series:[{name, values, color}], height, events} */
function stackChart(el, cfg) {
  const W = Math.max(280, el.clientWidth || 600), H = cfg.height || 220, n = cfg.x.length;
  const M = { l: 44, r: 10, t: 8, b: 26 }, iw = W - M.l - M.r, ih = H - M.t - M.b;
  const slot = iw / n, bw = Math.max(3, Math.min(34, slot * 0.72)), Y = v => M.t + ih - v * ih;
  const ev = (cfg.events || []);
  let s = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="category mix over time">`;
  [0, .25, .5, .75, 1].forEach(t => { s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)"/><text x="${M.l - 8}" y="${Y(t) + 3.5}" text-anchor="end">${t * 100}%</text>`; });
  const step = xLabelStep(n, iw);
  for (let i = 0; i < n; i++) {
    const x = M.l + slot * i + (slot - bw) / 2; let acc = 0;
    const tot = cfg.series.reduce((a, se) => a + (isNum(se.values[i]) ? se.values[i] : 0), 0) || 1;
    cfg.series.forEach(se => {
      const v = (isNum(se.values[i]) ? se.values[i] : 0) / tot; if (v <= 0) return;
      const y0 = Y(acc), y1 = Y(acc + v); acc += v;
      const h = Math.max(0, y0 - y1 - 2);
      s += `<rect x="${x}" y="${y1 + 1}" width="${bw}" height="${h}" fill="${se.color}" rx="${Math.min(2, h / 2)}"/>`;
    });
    if (i % step === 0) s += `<text x="${x + bw / 2}" y="${H - 8}" text-anchor="middle">${esc(shortPeriod(cfg.x[i]))}</text>`;
    s += `<rect class="hit" data-i="${i}" x="${M.l + slot * i}" y="${M.t}" width="${slot}" height="${ih}" fill="transparent"/>`;
  }
  s += `</svg>`;
  el.innerHTML = s;
  $$(".hit", el).forEach(h => {
    h.addEventListener("mousemove", e => {
      const i = +h.dataset.i; h.setAttribute("fill", "var(--s1-wash)");
      const rows = cfg.series.slice().reverse().map(se => ({ label: se.name, value: fmt.pct(se.values[i]), color: se.color }));
      ev.filter(x => x.i === i).forEach(x => rows.push({ note: `<b>◆ ${esc(x.label)}</b><br>${esc(x.text)}`, html: true }));
      showTip(e, tipRows(cfg.x[i], rows));
    });
    h.addEventListener("mouseleave", () => { h.setAttribute("fill", "transparent"); hideTip(); });
  });
}

/** SHAP dependence scatter (numeric x) with binned-mean trend. cfg: {points:[[x,y]], xLog, height} */
function scatterChart(el, cfg) {
  const W = Math.max(280, el.clientWidth || 600), H = cfg.height || 260;
  const M = { l: 56, r: 16, t: 10, b: 30 }, iw = W - M.l - M.r, ih = H - M.t - M.b;
  const pts = (cfg.points || []).filter(p => isNum(p[0]) && isNum(p[1]));
  if (!pts.length) { el.innerHTML = `<div class="empty">No SHAP sample.</div>`; return; }
  const tx = cfg.xLog ? (v => Math.sign(v) * Math.log10(1 + Math.abs(v))) : (v => v);
  const xs = pts.map(p => tx(p[0])), ys = pts.map(p => p[1]);
  let x0 = Math.min(...xs), x1 = Math.max(...xs); if (x0 === x1) { x0 -= 1; x1 += 1; }
  const xp = (x1 - x0) * 0.03; x0 -= xp; x1 += xp;
  const ytk = ticks(Math.min(0, ...ys), Math.max(0, ...ys), Math.max(3, Math.floor(ih / 44)));
  const y0 = ytk[0], y1 = ytk[ytk.length - 1];
  const X = t => M.l + (t - x0) / (x1 - x0) * iw, Y = v => M.t + ih - (v - y0) / (y1 - y0) * ih;
  let s = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="SHAP dependence">`;
  ytk.forEach(t => { s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="${t === 0 ? "var(--axis)" : "var(--grid)"}"/><text x="${M.l - 8}" y="${Y(t) + 3.5}" text-anchor="end">${t > 0 ? "+" : ""}${esc(fmt.num(t))}</text>`; });
  let xticks;
  if (cfg.xLog) {
    const raw = [0]; for (let p = 0; p <= 7; p++) { raw.push(Math.pow(10, p)); raw.push(-Math.pow(10, p)); }
    xticks = raw.map(v => [v, tx(v)]).filter(([, t]) => t >= x0 && t <= x1).sort((a, b) => a[1] - b[1]);
  } else xticks = ticks(x0, x1, Math.max(3, Math.floor(iw / 90))).filter(t => t >= x0 && t <= x1).map(t => [t, t]);
  xticks.forEach(([v, t]) => { s += `<text x="${X(t)}" y="${H - 10}" text-anchor="middle">${esc(fmt.num(v))}</text>`; });
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${M.t + ih}" y2="${M.t + ih}" stroke="var(--axis)"/>`;
  pts.forEach((p, i) => { s += `<circle cx="${X(xs[i]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="2.6" fill="var(--s1)" fill-opacity=".42"/>`; });
  const order = xs.map((x, i) => i).sort((a, b) => xs[a] - xs[b]), nb = Math.min(16, Math.max(4, Math.floor(pts.length / 15)));
  let d = "";
  for (let b = 0; b < nb; b++) {
    const ids = order.slice(Math.floor(b * order.length / nb), Math.floor((b + 1) * order.length / nb)); if (!ids.length) continue;
    d += (d ? "L" : "M") + X(mean(ids.map(i => xs[i]))).toFixed(1) + "," + Y(mean(ids.map(i => ys[i]))).toFixed(1);
  }
  s += `<path d="${d}" fill="none" stroke="var(--ink-2)" stroke-width="2" stroke-linejoin="round"/>`;
  s += `<circle class="hp" r="5" fill="none" stroke="var(--ink)" stroke-width="1.5" style="display:none"/>`;
  s += `<rect class="hit" x="${M.l}" y="${M.t}" width="${iw}" height="${ih}" fill="transparent"/></svg>`;
  el.innerHTML = s;
  const svg = el.firstChild, hp = svg.querySelector(".hp");
  svg.querySelector(".hit").addEventListener("mousemove", e => {
    const r = svg.getBoundingClientRect(), mx = (e.clientX - r.left) * (W / r.width), my = (e.clientY - r.top) * (H / r.height);
    let best = -1, bd = Infinity;
    pts.forEach((p, i) => { const dd = (X(xs[i]) - mx) ** 2 + (Y(p[1]) - my) ** 2; if (dd < bd) { bd = dd; best = i; } });
    if (best < 0 || bd > 900) { hp.style.display = "none"; hideTip(); return; }
    hp.style.display = ""; hp.setAttribute("cx", X(xs[best])); hp.setAttribute("cy", Y(pts[best][1]));
    showTip(e, tipRows("Transaction", [{ label: "feature value", value: fmt.num(pts[best][0]) }, { label: "SHAP (log-odds)", value: (pts[best][1] >= 0 ? "+" : "") + pts[best][1].toFixed(4) }]));
  });
  svg.querySelector(".hit").addEventListener("mouseleave", () => { hp.style.display = "none"; hideTip(); });
}

/** SHAP by category: jittered strip + mean tick. cfg: {points:[[cat,y]], cats} */
function stripChart(el, cfg) {
  const W = Math.max(280, el.clientWidth || 600), H = cfg.height || 260;
  const groups = {}; (cfg.points || []).forEach(([c, y]) => { if (isNum(y)) (groups[c] = groups[c] || []).push(y); });
  const cats = (cfg.cats || Object.keys(groups)).filter(c => groups[c]);
  if (!cats.length) { el.innerHTML = `<div class="empty">No SHAP sample.</div>`; return; }
  const all = [].concat(...cats.map(c => groups[c]));
  const M = { l: 56, r: 12, t: 10, b: 28 }, iw = W - M.l - M.r, ih = H - M.t - M.b;
  const ytk = ticks(Math.min(0, ...all), Math.max(0, ...all), 5), y0 = ytk[0], y1 = ytk[ytk.length - 1];
  const Y = v => M.t + ih - (v - y0) / (y1 - y0) * ih, slot = iw / cats.length;
  let s = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="SHAP by category">`;
  ytk.forEach(t => { s += `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(t)}" y2="${Y(t)}" stroke="${t === 0 ? "var(--axis)" : "var(--grid)"}"/><text x="${M.l - 8}" y="${Y(t) + 3.5}" text-anchor="end">${t > 0 ? "+" : ""}${esc(fmt.num(t))}</text>`; });
  cats.forEach((c, k) => {
    const cx = M.l + slot * k + slot / 2, w = Math.min(70, slot * 0.6);
    groups[c].forEach((y, j) => { const jit = ((j * 0.618034) % 1 - 0.5) * w; s += `<circle cx="${(cx + jit).toFixed(1)}" cy="${Y(y).toFixed(1)}" r="2.6" fill="var(--s1)" fill-opacity=".42"/>`; });
    const m = mean(groups[c]);
    s += `<line x1="${cx - w / 2 - 4}" x2="${cx + w / 2 + 4}" y1="${Y(m)}" y2="${Y(m)}" stroke="var(--ink)" stroke-width="2.5" stroke-linecap="round"/>`;
    s += `<text x="${cx}" y="${H - 9}" text-anchor="middle">${esc(c)}</text>`;
    s += `<rect class="hit" data-k="${k}" x="${M.l + slot * k}" y="${M.t}" width="${slot}" height="${ih}" fill="transparent"/>`;
  });
  s += `</svg>`;
  el.innerHTML = s;
  $$(".hit", el).forEach(h => {
    h.addEventListener("mousemove", e => {
      const c = cats[+h.dataset.k], g = groups[c].slice().sort((a, b) => a - b); h.setAttribute("fill", "var(--s1-wash)");
      showTip(e, tipRows(c, [{ label: "sampled txns", value: g.length }, { label: "mean SHAP", value: mean(g).toFixed(4) }, { label: "10th–90th pct", value: `${g[Math.floor(g.length * .1)].toFixed(3)} – ${g[Math.floor(g.length * .9)].toFixed(3)}` }]));
    });
    h.addEventListener("mouseleave", () => { h.setAttribute("fill", "transparent"); hideTip(); });
  });
}

/** Heatmap. cfg: {rows:[{key,label}], cols:[label], value(r,c), color(v)->[r,g,b]|null, fmt, square, maxCell, cellH,
 *  labelW, colLabels, showValues, tip(r,c)->html, onCell(r,c), onRow(r), selectedRow, outline(r,c)} */
function heatmap(el, cfg) {
  const W = Math.max(280, el.clientWidth || 600), nr = cfg.rows.length, nc = cfg.cols.length;
  const lw = cfg.labelW || 150, top = 4, bottom = cfg.colLabels ? 24 : 4;
  let cw = (W - lw - 8) / nc; if (cfg.maxCell) cw = Math.min(cw, cfg.maxCell);
  const chh = cfg.square ? cw : (cfg.cellH || 24), H = top + nr * chh + bottom, gw = lw + nc * cw + 4;
  let s = `<svg class="hm" viewBox="0 0 ${gw} ${H}" width="${gw}" height="${H}" style="max-width:100%" role="img" aria-label="${esc(cfg.aria || "heatmap")}">`;
  cfg.rows.forEach((row, r) => {
    const lab = row.label.length > (lw / 6.6) ? row.label.slice(0, Math.floor(lw / 6.6) - 1) + "…" : row.label;
    s += `<text class="${cfg.onRow ? "rowlab" : ""}${cfg.selectedRow === r ? " sel" : ""}" data-r="${r}" x="${lw - 8}" y="${top + r * chh + chh / 2 + 4}" text-anchor="end"${cfg.monoRows ? ' style="font-family:ui-monospace,monospace"' : ""}>${esc(lab)}</text>`;
    for (let c = 0; c < nc; c++) {
      const v = cfg.value(r, c), col = isNum(v) ? cfg.color(v) : null, x = lw + c * cw, y = top + r * chh;
      s += `<rect x="${x + .5}" y="${y + .5}" width="${Math.max(1, cw - 1)}" height="${Math.max(1, chh - 1)}" rx="2" fill="${col ? rgbStr(col) : "var(--surface-2)"}"${cfg.outline && cfg.outline(r, c) ? ' stroke="var(--ink)" stroke-width="1.5"' : ""}/>`;
      if (cfg.showValues && col && cw >= 34 && chh >= 16) s += `<text x="${x + cw / 2}" y="${y + chh / 2 + 3.5}" text-anchor="middle" style="fill:${inkOn(col)};font-size:10px">${esc(cfg.fmt(v))}</text>`;
    }
  });
  if (cfg.colLabels) { const st = xLabelStep(nc, nc * cw); cfg.cols.forEach((c, i) => { if (i % st === 0) s += `<text x="${lw + i * cw + cw / 2}" y="${H - 8}" text-anchor="middle">${esc(cfg.colLabels(c))}</text>`; }); }
  s += `<rect class="hov" fill="none" stroke="var(--ink)" stroke-width="2" rx="2" style="display:none" pointer-events="none"/>`;
  s += `<rect class="hit" x="${lw}" y="${top}" width="${nc * cw}" height="${nr * chh}" fill="transparent" style="cursor:${cfg.onCell ? "pointer" : "default"}"/></svg>`;
  el.innerHTML = s;
  const svg = el.firstChild, hov = svg.querySelector(".hov"), hit = svg.querySelector(".hit");
  const cell = e => { const b = svg.getBoundingClientRect(), sx = gw / b.width; const c = Math.floor(((e.clientX - b.left) * sx - lw) / cw), r = Math.floor(((e.clientY - b.top) * sx - top) / chh); return r >= 0 && r < nr && c >= 0 && c < nc ? [r, c] : null; };
  hit.addEventListener("mousemove", e => {
    const rc = cell(e); if (!rc) { hov.style.display = "none"; hideTip(); return; }
    hov.style.display = ""; hov.setAttribute("x", lw + rc[1] * cw + .5); hov.setAttribute("y", top + rc[0] * chh + .5); hov.setAttribute("width", cw - 1); hov.setAttribute("height", chh - 1);
    showTip(e, cfg.tip(rc[0], rc[1]));
  });
  hit.addEventListener("mouseleave", () => { hov.style.display = "none"; hideTip(); });
  if (cfg.onCell) hit.addEventListener("click", e => { const rc = cell(e); if (rc) cfg.onCell(rc[0], rc[1]); });
  if (cfg.onRow) $$(".rowlab", svg).forEach(t => t.addEventListener("click", () => cfg.onRow(+t.dataset.r)));
}

function sparkline(values, opts = {}) {
  const w = opts.w || 120, h = opts.h || 26, v = values.filter(isNum);
  if (v.length < 2) return "";
  let mn = Math.min(...v), mx = Math.max(...v); if (opts.min != null) mn = Math.min(mn, opts.min); if (mn === mx) mx = mn + 1;
  const X = i => 1 + i / (values.length - 1) * (w - 2), Y = x => h - 2 - (x - mn) / (mx - mn) * (h - 4);
  let d = "", pen = false; values.forEach((x, i) => { if (isNum(x)) { d += (pen ? "L" : "M") + X(i).toFixed(1) + "," + Y(x).toFixed(1); pen = true; } else pen = false; });
  const li = values.length - 1 - values.slice().reverse().findIndex(isNum);
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true"><path d="${d}" fill="none" stroke="${opts.color || "var(--s1)"}" stroke-width="1.5" vector-effect="non-scaling-stroke"/>` +
    `<circle cx="${X(li)}" cy="${Y(values[li])}" r="2" fill="${opts.color || "var(--s1)"}"/></svg>`;
}

/* ---------------------------------------------------------------- lineage graph */
const LG = (function build() {
  const nodes = new Map(), edges = [];
  const add = (id, kind, label, f) => { if (!nodes.has(id)) nodes.set(id, { id, kind, label, f, parents: new Set(), children: new Set() }); return nodes.get(id); };
  const link = (a, b) => { if (nodes.get(a).children.has(b)) return; nodes.get(a).children.add(b); nodes.get(b).parents.add(a); edges.push([a, b]); };
  FEATS.forEach(f => add("f:" + f.name, "feature", f.name, f));
  FEATS.forEach(f => {
    (f.source_tables || []).forEach(t => { add("t:" + t, "table", t); link("t:" + t, "f:" + f.name); });
    (f.upstream || []).forEach(u => { add("f:" + u, "feature", u, BYNAME[u] || null); link("f:" + u, "f:" + f.name); });
  });
  const memo = {}, visiting = new Set();
  const layer = id => {
    if (id in memo) return memo[id];
    const nd = nodes.get(id); if (nd.kind === "table") return (memo[id] = 0);
    if (visiting.has(id)) return 1;
    visiting.add(id); let l = 1;
    nd.parents.forEach(p => { if (nodes.get(p).kind === "feature") l = Math.max(l, layer(p) + 1); });
    visiting.delete(id); return (memo[id] = l);
  };
  nodes.forEach(n => { n.layer = layer(n.id); });
  return { nodes, edges };
})();
function closure(id, dir) {
  const out = new Set(), st = [id];
  while (st.length) { const n = LG.nodes.get(st.pop()); if (!n) continue; (dir === "up" ? n.parents : n.children).forEach(x => { if (!out.has(x)) { out.add(x); st.push(x); } }); }
  return out;
}
/** opts: {ids:Set|null, focus, selected, query, onSelect(id)} */
function renderLineage(el, opts) {
  const ids = new Set(opts.ids ? [...opts.ids] : [...LG.nodes.keys()]);
  const layers = [...new Set([...ids].map(id => LG.nodes.get(id).layer))].sort((a, b) => a - b);
  const cols = layers.map(l => [...ids].filter(id => LG.nodes.get(id).layer === l));
  const featKey = id => { const f = LG.nodes.get(id).f; return f ? [f.group, rank(f)] : ["~", 1e9]; };
  cols.forEach((c, ci) => c.sort(ci === 0 && LG.nodes.get(c[0]).kind === "table" ? ((a, b) => a.localeCompare(b)) : ((a, b) => { const x = featKey(a), y = featKey(b); return x[0].localeCompare(y[0]) || x[1] - y[1]; })));
  const pos = () => { const p = {}; cols.forEach(c => c.forEach((id, i) => { p[id] = i - c.length / 2; })); return p; };
  for (let it = 0; it < 6; it++) {
    let p = pos();
    for (let ci = 1; ci < cols.length; ci++) {
      const bc = id => { const ps = [...LG.nodes.get(id).parents].filter(x => ids.has(x)); return ps.length ? mean(ps.map(x => p[x])) : p[id]; };
      cols[ci].sort((a, b) => bc(a) - bc(b)); p = pos();
    }
    for (let ci = cols.length - 2; ci >= 0; ci--) {
      const bc = id => { const cs = [...LG.nodes.get(id).children].filter(x => ids.has(x)); return cs.length ? mean(cs.map(x => p[x])) : p[id]; };
      cols[ci].sort((a, b) => bc(a) - bc(b)); p = pos();
    }
  }
  const NW = 232, NH = 24, GY = 6, CG = 76, maxN = Math.max(...cols.map(c => c.length));
  const W = cols.length * (NW + CG) - CG + 20, H = maxN * (NH + GY) + 20, xy = {};
  cols.forEach((c, ci) => { const off = (maxN - c.length) * (NH + GY) / 2; c.forEach((id, i) => { xy[id] = [10 + ci * (NW + CG), 10 + off + i * (NH + GY)]; }); });
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="feature lineage graph">`;
  LG.edges.forEach(([a, b]) => {
    if (!ids.has(a) || !ids.has(b)) return;
    const [x1, y1] = xy[a], [x2, y2] = xy[b], sx = x1 + NW, sy = y1 + NH / 2, ty = y2 + NH / 2, dx = (x2 - sx) / 2;
    s += `<path class="lg-edge" data-a="${esc(a)}" data-b="${esc(b)}" d="M${sx},${sy}C${sx + dx},${sy} ${x2 - dx},${ty} ${x2},${ty}"/>`;
  });
  const q = (opts.query || "").toLowerCase();
  ids.forEach(id => {
    const n = LG.nodes.get(id), [x, y] = xy[id], f = n.f, sev = f ? maxSev(f) : 0;
    const stroke = sev === 3 ? "var(--critical)" : sev === 2 ? "var(--warning)" : "var(--border-strong)";
    const cls = ["lg-node", f && f.status === "deprecated" ? "dep" : "", opts.focus === id || opts.selected === id ? "focus" : "", q && n.label.toLowerCase().includes(q) ? "match" : ""].join(" ");
    const lab = n.label.length > 30 ? n.label.slice(0, 29) + "…" : n.label;
    s += `<g class="${cls}" data-id="${esc(id)}"><rect x="${x}" y="${y}" width="${NW}" height="${NH}" rx="6" fill="${n.kind === "table" ? "var(--surface-3)" : "var(--surface)"}" stroke="${stroke}"/>`;
    if (sev >= 2) s += `<path d="${sev === 3 ? `M${x + 10},${y + 6}l5,10h-10z` : ""}" fill="var(--critical)"/>` + (sev === 2 ? `<circle cx="${x + 10}" cy="${y + NH / 2}" r="4.5" fill="var(--warning)"/>` : "");
    s += `<text x="${x + (sev >= 2 ? 20 : 8)}" y="${y + 16}">${esc(lab)}</text>`;
    if (f) s += `<text class="r" x="${x + NW - 6}" y="${y + 16}" text-anchor="end">#${rank(f) < 1e9 ? rank(f) : "–"}</text>`;
    else if (n.kind === "table") s += `<text class="r" x="${x + NW - 6}" y="${y + 16}" text-anchor="end">${closure(id, "down").size}↓</text>`;
    s += `</g>`;
  });
  s += `</svg>`;
  el.innerHTML = s;
  const svg = el.firstChild;
  $$(".lg-node", svg).forEach(g => {
    const id = g.dataset.id, n = LG.nodes.get(id);
    g.addEventListener("mouseenter", e => {
      const on = new Set([id, ...closure(id, "up"), ...closure(id, "down")]);
      svg.classList.add("hl");
      $$(".lg-node", svg).forEach(x => x.classList.toggle("on", on.has(x.dataset.id)));
      $$(".lg-edge", svg).forEach(x => x.classList.toggle("on", on.has(x.dataset.a) && on.has(x.dataset.b)));
      const rows = n.f ? [{ label: "group", value: n.f.group }, { label: "importance rank", value: rank(n.f) < 1e9 ? "#" + rank(n.f) : "—" }, { label: "status", value: n.f.status }]
        : n.kind === "table" ? [{ label: "dependent features", value: [...closure(id, "down")].length }] : [{ label: "not in this catalog", value: "" }];
      (n.f && n.f.alerts || []).filter(a => a.level !== "info").forEach(a => rows.push({ note: (a.level === "critical" ? "▲ " : "● ") + a.message }));
      showTip(e, tipRows(n.label, rows));
    });
    g.addEventListener("mouseleave", () => { svg.classList.remove("hl"); hideTip(); });
    g.addEventListener("click", () => opts.onSelect && opts.onSelect(id));
  });
}

/* ---------------------------------------------------------------- syntax highlight */
const SQL_KW = "select|from|where|join|left|right|inner|outer|full|cross|on|and|or|not|as|case|when|then|else|end|group|by|order|having|over|partition|rows|range|between|preceding|following|current|row|interval|with|union|all|distinct|limit|is|null|in|exists|window|filter|cast|coalesce|nullif|true|false|date|day|hour|minute|second|zone|at|desc|asc|bigint|double|unbounded|lateral|using";
const PY_KW = "def|return|import|from|as|for|in|if|elif|else|while|with|lambda|not|and|or|is|None|True|False|class|try|except|finally|raise|yield|pass|break|continue|global|assert|async|await";
function highlight(code, lang) {
  const kw = new RegExp("^(?:" + (lang === "sql" ? SQL_KW : PY_KW) + ")$", lang === "sql" ? "i" : "");
  const re = lang === "sql"
    ? /(--[^\n]*|\/\*[\s\S]*?\*\/)|('(?:[^']|'')*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)(?=\s*\()|([A-Za-z_][A-Za-z0-9_]*)/g
    : /(#[^\n]*)|("{3}[\s\S]*?"{3}|'{3}[\s\S]*?'{3}|[rbuf]?"(?:\\.|[^"\\\n])*"|[rbuf]?'(?:\\.|[^'\\\n])*')|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)(?=\s*\()|([A-Za-z_][A-Za-z0-9_]*)/g;
  let out = "", last = 0, m;
  while ((m = re.exec(code))) {
    out += esc(code.slice(last, m.index)); last = re.lastIndex;
    const t = esc(m[0]);
    if (m[1]) out += `<span class="tk-com">${t}</span>`;
    else if (m[2]) out += `<span class="tk-str">${t}</span>`;
    else if (m[3]) out += `<span class="tk-num">${t}</span>`;
    else if (m[4]) out += kw.test(m[4]) ? `<span class="tk-kw">${t}</span>` : `<span class="tk-fn">${t}</span>`;
    else out += kw.test(m[5]) ? `<span class="tk-kw">${t}</span>` : t;
  }
  out += esc(code.slice(last));
  return out.split("\n").map((l, i) => `<span class="lnum">${i + 1}</span>${l}`).join("\n");
}

/* ---------------------------------------------------------------- state */
const SECTIONS = [
  { id: "overview", title: "Overview", sub: "Headline metrics vs. baseline" },
  { id: "model", title: "In-model behaviour", sub: "SHAP dependence · importance over time · drop-column impact" },
  { id: "definition", title: "Definition", sub: "Exact SQL / Python implementation" },
  { id: "metadata", title: "Metadata & lineage", sub: "Entity, window, sources, dependency graph" },
  { id: "performance", title: "Performance over time", sub: `AUC · KS · IV · recall and $ recall @${FPR_TXT} FPR` },
  { id: "stability", title: "Stability (PSI)", sub: "Population shift vs. baseline" },
  { id: "segments", title: "Segments", sub: "Drift and performance by channel / region" },
  { id: "distribution", title: "Distribution", sub: "Baseline vs. latest · fraud rate by bin" },
  { id: "drift", title: "Distribution over time", sub: "Quantile fan / category mix by period" },
  { id: "missing", title: "Missingness", sub: "Null rate by period" },
  { id: "correlations", title: "Correlated features", sub: "Redundancy check" },
  { id: "rules", title: "Alert rules", sub: "Thresholds in effect for this feature" },
  { id: "changelog", title: "Changelog & events", sub: "Versions, incidents and global events" },
  { id: "table", title: "Data table", sub: "Every timeline value, copyable" }
];
const DEFAULT_COLLAPSED = ["table", "rules"];
const S = {
  view: "catalog", current: null,
  q: "", groups: new Set(), status: "", dtype: "", alertsOnly: false, sort: store.get("sort", "rank"),
  hidden: new Set(store.get("hidden", [])), collapsed: new Set(store.get("collapsed", DEFAULT_COLLAPSED)),
  pinned: store.get("pinned", []).filter(n => BYNAME[n]),
  perfMetric: store.get("perfMetric", "auc"), codeLang: store.get("codeLang", "sql"), distMode: store.get("distMode", "shares"),
  catSort: { key: "rank", dir: 1 }, cmpMetric: store.get("cmpMetric", "auc"),
  sidebar: store.get("sidebar", true), showEvents: store.get("showEvents", true),
  segment: store.get("segment", ""), segDim: store.get("segDim", ""), segMetric: store.get("segMetric", "psi"),
  redThr: store.get("redThr", 0.8), redOrder: store.get("redOrder", "cluster"),
  lgSel: null, lgQuery: "", lgDeprecated: store.get("lgDeprecated", true), shapLog: store.get("shapLog", false)
};
if (!METRIC[S.perfMetric] || !METRIC[S.perfMetric].labels) S.perfMetric = "auc";
function persist() {
  ["perfMetric", "codeLang", "distMode", "sort", "cmpMetric", "sidebar", "showEvents", "segment", "segDim", "segMetric", "redThr", "redOrder", "lgDeprecated", "shapLog", "pinned"].forEach(k => store.set(k, S[k]));
  store.set("hidden", [...S.hidden]); store.set("collapsed", [...S.collapsed]);
}
recomputeAll();

/* ---------------------------------------------------------------- filtering */
function matches(f) {
  if (S.groups.size && !S.groups.has(f.group)) return false;
  if (S.status && f.status !== S.status) return false;
  if (S.dtype && f.dtype !== S.dtype) return false;
  if (S.alertsOnly && !(f.alerts || []).some(a => a.level !== "info")) return false;
  if (S.q) {
    const terms = S.q.toLowerCase().split(/\s+/).filter(Boolean);
    const hay = [f.name, f.group, f.entity, f.description, f.rationale, f.owner, (f.tags || []).join(" "),
      (f.source_tables || []).join(" "), f.code && f.code.sql, f.code && f.code.python].join(" ").toLowerCase();
    if (!terms.every(t => t.startsWith("tag:") ? (f.tags || []).some(x => x.toLowerCase() === t.slice(4)) : hay.includes(t))) return false;
  }
  return true;
}
function rank(f) { return (f && f.importance && isNum(f.importance.rank)) ? f.importance.rank : 1e9; }
function sorter(key) {
  const k = f => (f.kpis || {})[key];
  switch (key) {
    case "name": return (a, b) => a.name.localeCompare(b.name);
    case "group": return (a, b) => a.group.localeCompare(b.group) || rank(a) - rank(b);
    case "alerts": return (a, b) => maxSev(b) - maxSev(a) || (b.alerts || []).length - (a.alerts || []).length || rank(a) - rank(b);
    case "rank": return (a, b) => rank(a) - rank(b);
    default: return (a, b) => (isNum(k(b)) ? k(b) : -1e9) - (isNum(k(a)) ? k(a) : -1e9);
  }
}
function visible() { return FEATS.filter(matches).sort(sorter(S.sort)); }

/* ---------------------------------------------------------------- sidebar */
function initFilters() {
  const groups = [...new Set(FEATS.map(f => f.group))].sort();
  $("#groupChips").innerHTML = groups.map(g => `<button class="chip" data-g="${esc(g)}" aria-pressed="false">${esc(g)} <span class="hint">${FEATS.filter(f => f.group === g).length}</span></button>`).join("");
  $("#groupChips").addEventListener("click", e => {
    const b = e.target.closest(".chip"); if (!b) return;
    const g = b.dataset.g; S.groups.has(g) ? S.groups.delete(g) : S.groups.add(g); b.setAttribute("aria-pressed", S.groups.has(g)); refresh();
  });
  [...new Set(FEATS.map(f => f.status))].sort().forEach(s => $("#fStatus").insertAdjacentHTML("beforeend", `<option>${esc(s)}</option>`));
  [...new Set(FEATS.map(f => f.dtype))].sort().forEach(s => $("#fType").insertAdjacentHTML("beforeend", `<option>${esc(s)}</option>`));
  $("#q").addEventListener("input", e => { S.q = e.target.value.trim(); refresh(); });
  $("#fStatus").addEventListener("change", e => { S.status = e.target.value; refresh(); });
  $("#fType").addEventListener("change", e => { S.dtype = e.target.value; refresh(); });
  $("#fAlerts").addEventListener("change", e => { S.alertsOnly = e.target.checked; refresh(); });
  $("#sortBy").value = S.sort;
  $("#sortBy").addEventListener("change", e => { S.sort = e.target.value; persist(); refresh(); });
  $("#clearFilters").addEventListener("click", clearFilters);
}
function clearFilters() {
  S.q = ""; $("#q").value = ""; S.groups.clear(); $$("#groupChips .chip").forEach(c => c.setAttribute("aria-pressed", "false"));
  S.status = ""; $("#fStatus").value = ""; S.dtype = ""; $("#fType").value = ""; S.alertsOnly = false; $("#fAlerts").checked = false; refresh();
}
function renderList() {
  const list = visible();
  $("#listCount").textContent = `${list.length} of ${FEATS.length} features`;
  $("#flist").innerHTML = list.map(f => `
    <div class="fitem" role="option" data-name="${esc(f.name)}" aria-selected="${S.view === "detail" && S.current === f.name}">
      <div class="fname" title="${esc(f.name)}">${esc(f.name)}</div>
      <div class="fright">${alertBadge(f)}<button class="pin" data-pin="${esc(f.name)}" aria-pressed="${S.pinned.includes(f.name)}" title="Pin to compare">${ICON.pin}</button></div>
      <div class="fsub">#${rank(f) < 1e9 ? rank(f) : "–"} · ${esc(f.group)} · ${esc(f.dtype)}${f.status !== "production" ? " · " + esc(f.status) : ""}</div>
    </div>`).join("") || `<div class="empty" style="padding:12px">No features match. <button class="link" id="clr2">Clear filters</button></div>`;
  const c2 = $("#clr2"); if (c2) c2.onclick = clearFilters;
}
$("#flist").addEventListener("click", e => {
  const p = e.target.closest("[data-pin]"); if (p) { e.stopPropagation(); togglePin(p.dataset.pin); return; }
  const it = e.target.closest(".fitem"); if (it) openFeature(it.dataset.name);
});
function togglePin(name) {
  const i = S.pinned.indexOf(name);
  if (i >= 0) S.pinned.splice(i, 1);
  else { if (S.pinned.length >= 4) { toast("Compare holds up to 4 features — unpin one first"); return; } S.pinned.push(name); }
  persist(); refresh(); toast(i >= 0 ? `Unpinned ${name}` : `Pinned ${name} (${S.pinned.length}/4)`);
}
function compareSet(names) { S.pinned = names.slice(0, 4); persist(); setView("compare"); }

/* ---------------------------------------------------------------- routing */
const VIEWS = ["catalog", "detail", "compare", "lineage", "redundancy"];
function setHash() {
  const h = S.view === "detail" && S.current ? "#f=" + encodeURIComponent(S.current) : "#" + S.view;
  if (location.hash !== h) history.replaceState(null, "", h);
}
function readHash() {
  const h = decodeURIComponent(location.hash.slice(1));
  if (h.startsWith("f=") && BYNAME[h.slice(2)]) { S.view = "detail"; S.current = h.slice(2); }
  else if (VIEWS.includes(h)) S.view = h;
}
function openFeature(name, section, seg) {
  S.view = "detail"; S.current = name;
  if (seg) { S.segment = seg; S.segDim = seg.split("=")[0]; persist(); }
  if (section) { S.hidden.delete(section); S.collapsed.delete(section); persist(); }
  refresh();
  if (section) { const el = document.getElementById("sec-" + section); if (el) el.scrollIntoView({ block: "start" }); }
  else $("#main").scrollTop = 0;
}
function setView(v) { S.view = v; if (v === "detail" && !S.current) S.current = (visible()[0] || FEATS[0] || {}).name; refresh(); $("#main").scrollTop = 0; }
$$(".seg [data-view]").forEach(b => b.addEventListener("click", () => setView(b.dataset.view)));

/* ---------------------------------------------------------------- views */
function refresh() {
  hideTip();
  $$(".topbar .seg [data-view]").forEach(b => b.setAttribute("aria-pressed", b.dataset.view === S.view));
  $("#pinCount").textContent = S.pinned.length ? S.pinned.length : "";
  $("#layout").classList.toggle("no-sidebar", !S.sidebar);
  renderList();
  ({ catalog: renderCatalog, compare: renderCompare, lineage: renderLineageView, redundancy: renderRedundancy }[S.view] || renderDetail)();
  setHash();
  const sel = $(".fitem[aria-selected=true]"); if (sel) sel.scrollIntoView({ block: "nearest" });
}

/* ---- redundancy helpers (shared by catalog tile + redundancy view) */
function redundancyGroups(list, thr) {
  const names = list.map(f => f.name).filter(n => CIDX[n] != null), parent = {};
  names.forEach(n => { parent[n] = n; });
  const find = x => parent[x] === x ? x : (parent[x] = find(parent[x]));
  for (let i = 0; i < names.length; i++) for (let j = i + 1; j < names.length; j++) { const r = corr(names[i], names[j]); if (isNum(r) && Math.abs(r) >= thr) parent[find(names[i])] = find(names[j]); }
  const g = {}; names.forEach(n => { (g[find(n)] = g[find(n)] || []).push(n); });
  return Object.values(g).filter(x => x.length > 1).map(m => m.sort((a, b) => ((BYNAME[b].importance || {}).shap_mean_abs || 0) - ((BYNAME[a].importance || {}).shap_mean_abs || 0)))
    .sort((a, b) => b.length - a.length || rank(BYNAME[a[0]]) - rank(BYNAME[b[0]]));
}
function clusterOrder(names) {
  const n = names.length; if (n < 3) return names.slice();
  let clusters = names.map((x, i) => ({ items: [i], id: i })), D = {};
  const key = (a, b) => a < b ? a + "," + b : b + "," + a;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const r = corr(names[i], names[j]); D[key(i, j)] = 1 - (isNum(r) ? Math.abs(r) : 0); }
  let next = n;
  while (clusters.length > 1) {
    let best = Infinity, bi = 0, bj = 1;
    for (let i = 0; i < clusters.length; i++) for (let j = i + 1; j < clusters.length; j++) { const d = D[key(clusters[i].id, clusters[j].id)]; if (d < best) { best = d; bi = i; bj = j; } }
    const A = clusters[bi], B = clusters[bj], C = { items: A.items.concat(B.items), id: next++ };
    clusters.forEach(K => { if (K !== A && K !== B) D[key(K.id, C.id)] = (A.items.length * D[key(K.id, A.id)] + B.items.length * D[key(K.id, B.id)]) / C.items.length; });
    clusters = clusters.filter(K => K !== A && K !== B); clusters.push(C);
  }
  return clusters[0].items.map(i => names[i]);
}

/* ---- catalog */
const CAT_COLS = [
  { key: "name", label: "Feature", l: true, get: f => f.name },
  { key: "group", label: "Group", l: true, get: f => f.group },
  { key: "status", label: "Status", l: true, get: f => f.status },
  { key: "rank", label: "Rank", get: f => rank(f) },
  { key: "shap", label: "Mean |SHAP|", get: f => f.importance && f.importance.shap_mean_abs },
  { key: "usd_at_risk", label: "$ at risk / mo", get: f => f.kpis.usd_at_risk, help: "Fraud $ per month the model would stop catching if this feature were dropped (drop-column retrain)." },
  { key: "auc", label: "AUC", get: f => f.kpis.auc },
  { key: "auc_d", label: "Δ AUC", get: f => isNum(f.kpis.auc) && isNum(f.kpis.auc_base) ? f.kpis.auc - f.kpis.auc_base : null },
  { key: "recall", label: `Recall @${FPR_TXT}`, get: f => f.kpis.recall_at_1pct_fpr },
  { key: "iv", label: "IV", get: f => f.kpis.iv },
  { key: "psi", label: "PSI", get: f => f.kpis.psi },
  { key: "seg_psi", label: "Worst seg. PSI", get: f => f.kpis.seg_psi },
  { key: "missing_rate", label: "Missing", get: f => f.kpis.missing_rate },
  { key: "alerts", label: "Alerts", get: f => maxSev(f) * 10 + (f.alerts || []).length }
];
function renderCatalog() {
  const list = visible();
  const ck = S.catSort, col = CAT_COLS.find(c => c.key === ck.key) || CAT_COLS[3];
  list.sort((a, b) => {
    const x = col.get(a), y = col.get(b);
    if (typeof x === "string" || typeof y === "string") return String(x).localeCompare(String(y)) * ck.dir;
    return ((isNum(x) ? x : -1e9) - (isNum(y) ? y : -1e9)) * ck.dir;
  });
  const maxShap = Math.max(...FEATS.map(f => (f.importance && f.importance.shap_mean_abs) || 0), 1e-9);
  const nCrit = FEATS.filter(f => maxSev(f) === 3).length, nWarn = FEATS.filter(f => maxSev(f) === 2).length;
  const nHidden = FEATS.filter(f => (f.alerts || []).some(a => a.metric === "segment")).length;
  const groups = CM ? redundancyGroups(FEATS, S.redThr) : [];
  const atRisk = FEATS.filter(f => (f.alerts || []).some(a => a.level === "critical")).reduce((a, f) => a + (f.kpis.usd_at_risk || 0), 0);
  const tiles = [
    { k: "Features", v: FEATS.length, d: Object.entries(FEATS.reduce((a, f) => (a[f.status] = (a[f.status] || 0) + 1, a), {})).map(([k, v]) => `${v} ${k}`).join(" · ") },
    { k: "Critical alerts", v: nCrit, d: atRisk ? `${fmt.usd(atRisk)}/mo of catch relies on them` : "features", ic: nCrit ? ICON.critical : "", click: "crit" },
    { k: "Warnings", v: nWarn, d: "features", ic: nWarn ? ICON.warning : "", click: "warn" },
    { k: "Hidden segment drift", v: nHidden, d: "stable overall, drifting in a segment", click: "seg" },
    { k: `Redundant groups (|r| ≥ ${S.redThr.toFixed(2)})`, v: CM ? groups.length : "—", d: CM ? `${groups.reduce((a, g) => a + g.length - 1, 0)} features to review` : "no correlation matrix", click: "red" },
    { k: "Periods", v: PERIODS.length || "—", d: PERIODS.length ? `${PERIODS[0]} → ${PERIODS[PERIODS.length - 1]} · baseline first ${NBASE}${MAT ? ` · last ${MAT} labels maturing` : ""}` : "" }
  ];
  const head = CAT_COLS.map(c => `<th class="${c.l ? "l" : ""}" data-k="${c.key}" title="${esc(c.help || "")}" aria-sort="${ck.key === c.key ? (ck.dir > 0 ? "ascending" : "descending") : "none"}">${esc(c.label)}${ck.key === c.key ? ` <span class="arrow">${ck.dir > 0 ? "▲" : "▼"}</span>` : ""}</th>`).join("") + `<th class="l">Trend (AUC · PSI · missing)</th><th></th>`;
  const rules = RULES;
  const rows = list.map(f => {
    const k = f.kpis, shap = f.importance && f.importance.shap_mean_abs, tl = f.timeline || {}, r = rulesFor(f);
    const dA = isNum(k.auc) && isNum(k.auc_base) ? k.auc - k.auc_base : null;
    const lv = (v, rule) => !rule || !isNum(v) ? "" : v >= rule.critical ? "cell-bad" : v >= rule.warning ? "cell-warn" : "";
    const psiCls = lv(k.psi, r.psi), segCls = lv(k.seg_psi, r.segment_psi);
    const missCls = isNum(k.missing_rate) && isNum(k.missing_rate_base) ? lv(k.missing_rate - k.missing_rate_base, r.missing_increase) : "";
    const aucCls = isNum(dA) ? lv(-dA, r.auc_drop) : "";
    const recCls = isNum(k.recall_at_1pct_fpr) && isNum(k.recall_at_1pct_fpr_base) ? lv(k.recall_at_1pct_fpr_base - k.recall_at_1pct_fpr, r.recall_drop) : "";
    return `<tr data-name="${esc(f.name)}">
      <td class="l"><span class="fname">${esc(f.name)}</span></td>
      <td class="l">${esc(f.group)}</td><td class="l">${statusBadge(f.status)}</td>
      <td class="num">${rank(f) < 1e9 ? rank(f) : "—"}</td>
      <td class="num">${isNum(shap) ? shap.toFixed(3) : "—"}<span class="shapbar" style="width:${isNum(shap) ? Math.max(1, shap / maxShap * 50) : 0}px"></span></td>
      <td class="num">${fmt.usd(k.usd_at_risk)}</td>
      <td class="num ${aucCls}">${fmt.auc(k.auc)}</td>
      <td class="num ${aucCls}">${signed(dA)}</td>
      <td class="num ${recCls}">${fmt.pct(k.recall_at_1pct_fpr)}</td>
      <td class="num">${fmt.auc(k.iv)}</td>
      <td class="num ${psiCls}">${fmt.auc(k.psi)}</td>
      <td class="num ${segCls}" title="${esc(k.seg_key || "")}">${fmt.auc(k.seg_psi)}</td>
      <td class="num ${missCls}">${fmt.pct(k.missing_rate)}</td>
      <td>${alertBadge(f) || '<span class="hint">—</span>'}</td>
      <td class="l">${sparkline(tl.auc || [], { w: 64, h: 20 })}${sparkline(tl.psi || [], { w: 64, h: 20, color: "var(--s2)", min: 0 })}${sparkline(tl.missing_rate || [], { w: 64, h: 20, color: "var(--s3)", min: 0 })}</td>
      <td><button class="pin" data-pin="${esc(f.name)}" aria-pressed="${S.pinned.includes(f.name)}" title="Pin to compare">${ICON.pin}</button></td>
    </tr>`;
  }).join("");
  $("#main").innerHTML = `<div class="catalog">
    <div class="summary-tiles">${tiles.map(t => `<div class="tile"${t.click ? ` data-click="${t.click}" style="cursor:pointer"` : ""}><div class="k"><span>${esc(t.k)}</span>${t.ic ? `<span class="alert-ic">${t.ic}</span>` : ""}</div><div class="v">${esc(t.v)}</div><div class="d">${esc(t.d)}</div></div>`).join("")}</div>
    <div class="ctl-row"><span class="hint">${list.length} shown · click a column to sort · click a row to open${MAT ? ` · label-based columns use ${esc(PERIODS[MATURE_IDX] || "")} (last period with mature labels)` : ""} · trends:</span>
      ${legendHTML([{ kind: "line", label: "AUC", color: "var(--s1)" }, { kind: "line", label: "PSI", color: "var(--s2)" }, { kind: "line", label: "Missing", color: "var(--s3)" }], "margin:0")}
      <span style="flex:1"></span><button class="btn small" id="exportCsv">Export CSV</button></div>
    <div class="ctable-wrap"><table class="ctable"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>
    ${MODEL.note ? `<p class="hint" style="margin-top:10px">${esc(MODEL.note)}</p>` : ""}
  </div>`;
  $$(".ctable th[data-k]").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.k; S.catSort = { key: k, dir: S.catSort.key === k ? -S.catSort.dir : (["name", "group", "status", "rank"].includes(k) ? 1 : -1) }; renderCatalog();
  }));
  $(".ctable tbody").addEventListener("click", e => {
    const p = e.target.closest("[data-pin]"); if (p) { e.stopPropagation(); togglePin(p.dataset.pin); return; }
    const tr = e.target.closest("tr[data-name]"); if (tr) openFeature(tr.dataset.name);
  });
  $$("[data-click]").forEach(t => t.addEventListener("click", () => {
    const c = t.dataset.click;
    if (c === "red") return setView("redundancy");
    if (c === "seg") { S.catSort = { key: "seg_psi", dir: -1 }; return renderCatalog(); }
    S.alertsOnly = true; $("#fAlerts").checked = true; S.catSort = { key: "alerts", dir: -1 }; refresh();
  }));
  $("#exportCsv").onclick = () => {
    const lines = [CAT_COLS.map(c => c.label).join(",")].concat(list.map(f => CAT_COLS.map(c => { const v = c.get(f); return typeof v === "string" ? `"${v.replace(/"/g, '""')}"` : (isNum(v) && v < 1e9 ? +v.toFixed(5) : ""); }).join(",")));
    const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" })); a.download = "features.csv"; a.click();
  };
}

/* ---- detail */
function delta(cur, base, metric) {
  const M = METRIC[metric], f = M.fmt;
  if (!isNum(cur) || !isNum(base)) return `<div class="d">no baseline</div>`;
  const d = cur - base;
  if (Math.abs(d) < 0.0005) return `<div class="d">≈ baseline (${f(base)})</div>`;
  const up = d > 0, good = M.better === "up" ? up : !up;
  return `<div class="d ${up ? "up" : "down"}-${good ? "good" : "bad"}">${up ? "▲" : "▼"} ${M.pp ? Math.abs(d * 100).toFixed(1) + "pp" : Math.abs(d).toFixed(3)} vs baseline ${f(base)}</div>`;
}
function card(id, title, sub, body) {
  if (S.hidden.has(id)) return "";
  return `<section class="card${S.collapsed.has(id) ? " collapsed" : ""}" id="sec-${id}" data-sec="${id}">
    <header role="button" tabindex="0" aria-expanded="${!S.collapsed.has(id)}">${ICON.chev}<h3>${esc(title)}</h3><span class="sub">${esc(sub)}</span>
      <span class="tools"><button class="btn ghost small" data-hide="${id}" title="Hide this section (restore via Sections menu)">${ICON.eyeOff}</button></span></header>
    <div class="body">${body}</div></section>`;
}
function segOptions(f) {
  const opts = [];
  Object.entries((f && f.segments) || FEATS.reduce((a, g) => Object.assign(a, g.segments || {}), {})).forEach(([dim, vals]) => Object.keys(vals).forEach(v => opts.push(dim + "=" + v)));
  return opts;
}
function segData(f, key) { if (!key) return null; const [dim, val] = key.split("="); return ((f.segments || {})[dim] || {})[val] || null; }
function chartHost(el, legendItems) { el.innerHTML = (legendItems && legendItems.length ? legendHTML(legendItems) : "") + `<div class="chart"></div>`; return el.lastChild; }

function renderDetail() {
  const f = BYNAME[S.current];
  if (!f) { $("#main").innerHTML = `<div class="detail"><div class="empty">Pick a feature from the list.</div></div>`; return; }
  const list = visible(), idx = list.findIndex(x => x.name === f.name);
  const k = f.kpis || {}, tl = f.timeline || {}, imp = f.importance || {}, mb = f.model_behavior || {}, ab = mb.ablation || {};
  const per = tl.periods || [];
  const asOf = m => METRIC[m].labels && MAT && k[m + "_idx"] >= 0 ? ` <span class="hint" title="Last period with mature labels">· ${esc(per[k[m + "_idx"]] || "")}</span>` : "";
  const tileFor = m => ({
    k: METRIC[m].short || METRIC[m].label, v: METRIC[m].fmt(k[m]), help: METRIC[m].help + (METRIC[m].labels && MAT ? " Uses the last period with mature labels." : ""),
    sec: m === "psi" ? "stability" : m === "missing_rate" ? "missing" : "performance", metric: m, asOf: asOf(m),
    d: m === "psi" ? `<div class="d">${isNum(k.psi_max6) ? "max last 6: " + fmt.auc(k.psi_max6) : ""}</div>` : delta(k[m], k[m + "_base"], m),
    spark: sparkline(tl[m] || [], { color: m === "psi" ? "var(--s2)" : m === "missing_rate" ? "var(--s3)" : "var(--s1)", min: m === "psi" || m === "missing_rate" ? 0 : undefined })
  });
  const tiles = [
    { k: "Importance rank", v: rank(f) < 1e9 ? "#" + rank(f) : "—", sec: "model", d: `<div class="d">mean |SHAP| ${isNum(imp.shap_mean_abs) ? imp.shap_mean_abs.toFixed(4) : "—"}</div>` },
    { k: "$ at risk if dropped", v: fmt.usd(ab.usd_at_risk_per_month), sec: "model", help: "Fraud dollars per month the model stops catching when retrained without this feature.", d: `<div class="d">per month · ΔAUC ${isNum(ab.delta_auc) ? signed(ab.delta_auc, 4) : "—"} if removed</div>` },
    ...["auc", "recall_at_1pct_fpr", "value_recall_at_1pct_fpr", "iv", "psi", "missing_rate"].map(tileFor)
  ];
  const alerts = (f.alerts || []).slice().sort((a, b) => SEV[b.level] - SEV[a.level]);
  const sm = f.summary || {};
  const summaryRows = Object.entries(sm).map(([a, b]) => `<dt>${esc(a.replace(/_/g, " "))}</dt><dd class="num">${typeof b === "number" ? (a.includes("rate") || a.includes("share") ? fmt.pct(b) : Number.isInteger(b) ? fmt.int(b) : fmt.num(b)) : esc(b)}</dd>`).join("");

  const overview = `
    <div class="tiles">${tiles.map(t => `<div class="tile" ${t.sec ? `data-jump="${t.sec}"${t.metric && METRIC[t.metric].labels ? ` data-metric="${t.metric}"` : ""} style="cursor:pointer"` : ""} title="${esc(t.help || "")}"><div class="k"><span>${esc(t.k)}${t.asOf || ""}</span></div><div class="v num">${esc(t.v)}</div>${t.d}${t.spark || ""}</div>`).join("")}</div>
    ${MAT && isNum(k.auc_naive) && isNum(k.auc) && Math.abs(k.auc_naive - k.auc) >= 0.02 ? `<p class="help"><b>Note:</b> the raw latest-period AUC is ${fmt.auc(k.auc_naive)}, but ${MAT} recent period${MAT > 1 ? "s have" : " has"} immature labels, so the headline uses ${esc(per[k.auc_idx])} (${fmt.auc(k.auc)}).</p>` : ""}
    ${summaryRows ? `<div class="subhead">Summary statistics (latest period${per.length ? ", " + esc(per[per.length - 1]) : ""})</div><dl class="kv" style="grid-template-columns:repeat(auto-fill,minmax(90px,max-content) minmax(70px,1fr))">${summaryRows}</dl>` : ""}`;

  const hasShap = (mb.shap_points || []).length;
  const model = mb.importance_timeline || hasShap || ab.delta_auc != null ? `
    <div class="tiles">
      <div class="tile"><div class="k"><span>Mean |SHAP|</span></div><div class="v num">${isNum(imp.shap_mean_abs) ? imp.shap_mean_abs.toFixed(4) : "—"}</div><div class="d">rank #${rank(f) < 1e9 ? rank(f) : "—"} of ${FEATS.length}${isNum(imp.gain) ? " · gain " + imp.gain.toFixed(4) : ""}</div></div>
      <div class="tile"><div class="k"><span>Drop-column ΔAUC</span></div><div class="v num">${isNum(ab.delta_auc) ? signed(ab.delta_auc, 4) : "—"}</div><div class="d">model AUC if retrained without it</div></div>
      <div class="tile"><div class="k"><span>Δ $ recall @${FPR_TXT}</span></div><div class="v num">${isNum(ab.delta_value_recall) ? signed(ab.delta_value_recall, 1, true) : "—"}</div><div class="d">fraud dollars caught</div></div>
      <div class="tile"><div class="k"><span>$ at risk / month</span></div><div class="v num">${fmt.usd(ab.usd_at_risk_per_month)}</div><div class="d">${MODEL.monthly_fraud_value_usd ? "of " + fmt.usd(MODEL.monthly_fraud_value_usd) + " monthly fraud" : ""}</div></div>
    </div>
    ${ab.method ? `<p class="help">Ablation: ${esc(ab.method)}. Low impact despite high SHAP usually means a correlated feature takes over — see Correlated features.</p>` : ""}
    <div class="grid2">
      <div><div class="ctl-row"><span class="subhead" style="margin:0">SHAP dependence</span>${f.dtype === "numeric" ? `<label class="toggle"><input type="checkbox" id="shapLog" ${S.shapLog ? "checked" : ""}> Log x</label>` : ""}<span class="hint">${isNum(mb.shap_missing_mean) ? `missing → mean SHAP ${mb.shap_missing_mean >= 0 ? "+" : ""}${mb.shap_missing_mean.toFixed(3)}` : ""}</span></div>
        ${legendHTML(f.dtype === "numeric" ? [{ label: "sampled transactions", color: "var(--s1)" }, { kind: "line", label: "binned mean", color: "var(--ink-2)" }] : [{ label: "sampled transactions", color: "var(--s1)" }, { kind: "line", label: "category mean", color: "var(--ink)" }])}
        <div class="chart" id="chShap"></div></div>
      <div><div class="subhead">Importance over time (mean |SHAP|)</div>${legendHTML(S.showEvents ? [{ kind: "event", label: "event" }] : [])}<div class="chart" id="chImp"></div></div>
    </div>` : `<div class="empty">No in-model data (provide model_behavior).</div>`;

  const code = f.code || {};
  const langs = ["sql", "python"].filter(l => code[l]);
  Object.keys(code).forEach(l => { if (!langs.includes(l) && code[l]) langs.push(l); });
  const lang = langs.includes(S.codeLang) ? S.codeLang : langs[0];
  const definition = langs.length ? `
    <div class="ctl-row"><div class="seg" id="langSeg">${langs.map(l => `<button data-lang="${l}" aria-pressed="${l === lang}">${l === "sql" ? "SQL" : l === "python" ? "Python" : esc(l)}</button>`).join("")}</div></div>
    <div class="codewrap"><pre id="codeBlock"><code>${highlight(code[lang], lang === "sql" ? "sql" : "python")}</code></pre>
      <button class="btn small copy" id="copyCode">${ICON.copy}Copy</button></div>` : `<div class="empty">No implementation recorded.</div>`;

  const lin = (arr, clickable) => (arr && arr.length) ? arr.map(x => clickable && BYNAME[x] ? `<button class="link" data-open="${esc(x)}">${esc(x)}</button>` : `<code>${esc(x)}</code>`).join(", ") : `<span class="hint">none</span>`;
  const downstream = FEATS.filter(g => (g.upstream || []).includes(f.name)).map(g => g.name);
  const metadata = `<dl class="kv">
    <dt>Group</dt><dd>${esc(f.group)}</dd>
    <dt>Entity / grain</dt><dd>${esc(f.entity || "—")}</dd>
    <dt>Data type</dt><dd>${esc(f.dtype)}</dd>
    <dt>Window</dt><dd>${esc(f.window || "—")}</dd>
    <dt>Freshness</dt><dd>${esc(f.freshness || "—")}</dd>
    <dt>Default / null meaning</dt><dd>${esc(f.default_value || "—")}</dd>
    <dt>Owner</dt><dd>${esc(f.owner || "—")}</dd>
    <dt>Version</dt><dd>${esc(f.version || "—")}${f.created ? ` <span class="hint">· created ${esc(f.created)}</span>` : ""}</dd>
    <dt>Source tables</dt><dd>${lin(f.source_tables, false)}</dd>
    <dt>Upstream features</dt><dd>${lin(f.upstream, true)}</dd>
    <dt>Downstream features</dt><dd>${lin(downstream, true)}</dd>
  </dl><div class="ctl-row" style="margin-top:14px"><span class="subhead" style="margin:0">Dependency graph</span><button class="btn small" id="openLineage">${ICON.graph}Full lineage</button><span class="hint">hover to trace · click a node to open it</span></div><div class="lg-wrap" id="miniLineage"></div>`;

  const segKey = S.segment && segData(f, S.segment) ? S.segment : "";
  const segBanner = segKey ? `<div class="seg-banner">Showing segment <b>${esc(segKey.replace("=", " = "))}</b> vs. all transactions (dashed). <button class="link" data-clearseg>Show all</button></div>` : "";
  const performance = per.length ? `${segBanner}
    <div class="ctl-row"><div class="seg" id="perfSeg">${LABEL_METRICS.map(m => `<button data-m="${m}" aria-pressed="${S.perfMetric === m}">${esc(METRIC[m].short || METRIC[m].label)}</button>`).join("")}</div>
      <span class="hint" id="perfHelp">${esc(METRIC[S.perfMetric].help)}</span></div>
    <div id="chPerf"></div>` : `<div class="empty">No timeline.</div>`;
  const stability = (tl.psi || []).length ? `${segBanner}<p class="help">${esc(METRIC.psi.help)}</p><div id="chPsi"></div>` : `<div class="empty">No PSI data.</div>`;

  const dims = Object.keys(f.segments || {});
  const sdim = dims.includes(S.segDim) ? S.segDim : dims[0];
  const segments = dims.length ? `
    <div class="ctl-row"><div class="seg" id="segDimSeg">${dims.map(d => `<button data-d="${esc(d)}" aria-pressed="${d === sdim}">${esc(d)}</button>`).join("")}</div>
      <div class="seg" id="segMetSeg">${["psi", "auc", "recall_at_1pct_fpr", "missing_rate", "share"].map(m => `<button data-m="${m}" aria-pressed="${S.segMetric === m}">${esc(m === "auc" ? "Δ AUC" : m === "recall_at_1pct_fpr" ? "Δ Recall" : METRIC[m].short || METRIC[m].label)}</button>`).join("")}</div>
      <span id="segLegend"></span></div>
    <p class="help">Rows are segments, columns are periods. Click a segment name to overlay it on the Performance, Stability and Missingness charts.</p>
    <div id="chSeg" style="overflow-x:auto"></div>` : `<div class="empty">No segment data (provide segments).</div>`;

  const dist = f.distribution || {};
  const distribution = (dist.bins || []).length ? `
    <div class="ctl-row"><div class="seg" id="distSeg">
      <button data-d="shares" aria-pressed="${S.distMode === "shares"}">Baseline vs latest</button>
      <button data-d="fraud" aria-pressed="${S.distMode === "fraud"}">Fraud rate by bin</button>
      <button data-d="both" aria-pressed="${S.distMode === "both"}">Both</button></div>
      <span class="hint">${f.dtype === "numeric" ? "Bins = baseline deciles" : "Categories"}${dist.bins.includes("Missing") ? " + missing bin" : ""}</span></div>
    <div id="distCharts"></div>` : `<div class="empty">No distribution data.</div>`;
  const hasQ = tl.quantiles && Object.keys(tl.quantiles).length, hasC = tl.category_shares && Object.keys(tl.category_shares).length;
  const drift = hasQ || hasC ? `<p class="help">${hasQ ? "Median with 25–75th (dark) and 5–95th (light) percentile bands, non-null values only." : "Share of each category among non-null values, per period."}</p>
    ${hasQ ? `<div class="ctl-row"><label class="toggle"><input type="checkbox" id="logY" ${store.get("logY", false) ? "checked" : ""}> Log scale</label><label class="toggle"><input type="checkbox" id="showMean" ${store.get("showMean", false) ? "checked" : ""}> Show mean</label></div>` : ""}
    <div id="chDrift"></div>` : `<div class="empty">No per-period distribution data.</div>`;
  const missing = (tl.missing_rate || []).length ? `${segBanner}<p class="help">${esc(f.default_value ? "Null meaning: " + f.default_value : "")}</p><div id="chMiss"></div>` : `<div class="empty">No missingness data.</div>`;
  const corrs = f.correlations || [];
  const correlations = corrs.length ? `<p class="help">Highest absolute correlation with other features. |r| ≥ ${S.redThr.toFixed(2)} is flagged as redundant (threshold set in the Redundancy view).</p><div class="corr">${corrs.map(c => `
      <div>${BYNAME[c.feature] ? `<button class="link" data-open="${esc(c.feature)}">${esc(c.feature)}</button>` : `<code>${esc(c.feature)}</code>`}${Math.abs(c.r) >= S.redThr ? ' <span class="review">redundant?</span>' : ""}</div>
      <div class="bar" title="r = ${c.r}"><i class="${c.r < 0 ? "neg" : ""}" style="${c.r < 0 ? `right:50%;width:${Math.abs(c.r) * 50}%` : `left:50%;width:${c.r * 50}%`}"></i></div>
      <div class="num" style="text-align:right">${c.r >= 0 ? "+" : "−"}${Math.abs(c.r).toFixed(2)}</div>`).join("")}</div>
      ${legendHTML([{ label: "positive", color: "var(--s1)" }, { label: "negative", color: "var(--s2)" }], "margin-top:10px")}
      <div class="ctl-row" style="margin-top:6px"><button class="btn small" id="cmpCorr">Compare with top correlate</button><button class="btn small" id="goRed">Open redundancy view</button></div>` : `<div class="empty">No correlation data.</div>`;

  const r = rulesFor(f), o = f.alert_overrides || {};
  const srcOf = (key, field) => o[key] && o[key][field] != null ? `<span class="src feature">feature override</span>` : (RULES[key] && RULES_FILE[key] && RULES[key][field] !== RULES_FILE[key][field]) ? `<span class="src edited">edited here</span>` : `<span class="src">default</span>`;
  const curVal = key => {
    if (key === "psi") return fmt.auc(k.psi);
    if (key === "missing_increase") return isNum(k.missing_rate) && isNum(k.missing_rate_base) ? signed(k.missing_rate - k.missing_rate_base, 1, true) : "—";
    if (key === "missing_spike") return isNum(k.missing_max) && isNum(k.missing_rate_base) ? signed(k.missing_max - k.missing_rate_base, 1, true) : "—";
    if (key === "auc_drop") return isNum(k.auc) && isNum(k.auc_base) ? (k.auc_base - k.auc).toFixed(3) : "—";
    if (key === "recall_drop") return isNum(k.recall_at_1pct_fpr) && isNum(k.recall_at_1pct_fpr_base) ? ((k.recall_at_1pct_fpr_base - k.recall_at_1pct_fpr) * 100).toFixed(1) + "pp" : "—";
    if (key === "segment_psi") return isNum(k.seg_psi) ? `${fmt.auc(k.seg_psi)} (${esc(k.seg_key)})` : "—";
    return "—";
  };
  const fired = key => (f.alerts || []).find(a => ({ psi: "psi", missing_increase: "missing_rate", missing_spike: "missing_rate", auc_drop: "auc", recall_drop: "recall_at_1pct_fpr", segment_psi: "segment" })[key] === a.metric);
  const rulesBody = `
    ${o.note ? `<p class="help"><b>Override note:</b> ${esc(o.note)}</p>` : ""}
    <div class="tablewrap" style="max-height:none"><table class="rules-tbl"><thead><tr><th>Rule</th><th>Warning</th><th>Critical</th><th>Current</th><th>Status</th></tr></thead><tbody>
    ${Object.keys(RULE_META).map(key => { const rr = r[key] || {}, off = (o.disabled || []).includes(key), fa = fired(key);
      return `<tr><td>${esc(RULE_META[key].label)}</td><td>${isNum(rr.warning) ? rr.warning : "—"} ${isNum(rr.warning) ? srcOf(key, "warning") : ""}</td><td>${isNum(rr.critical) ? rr.critical : "—"} ${isNum(rr.critical) ? srcOf(key, "critical") : ""}</td>
        <td class="num">${curVal(key)}</td><td>${off ? '<span class="hint">disabled for this feature</span>' : f.muted ? '<span class="hint">muted</span>' : fa ? `<span class="alert-ic">${ICON[fa.level]}<span class="t">${fa.level}</span></span>` : '<span class="ok-text">✓ ok</span>'}</td></tr>`; }).join("")}
    </tbody></table></div>
    <div class="ctl-row" style="margin-top:10px"><label class="toggle"><input type="checkbox" id="muteLocal" ${MUTED.has(f.name) ? "checked" : ""} ${o.muted ? "disabled" : ""}> Mute alerts for this feature (this browser only)</label>
      <button class="btn small" id="copyOverride">${ICON.copy}Copy override JSON</button><span class="hint">Edit global thresholds from the Rules menu.</span></div>`;

  const evs = eventsFor(f).slice().sort((a, b) => String(b.date).localeCompare(String(a.date)));
  const cl = (f.changelog || []).slice().reverse();
  const changelog = `<div class="grid2"><div><div class="subhead">Feature changelog</div>${cl.length ? `<ul class="timeline-list">${cl.map(c => `<li><div class="when">${esc(c.date)}${c.version ? " · v" + esc(c.version) : ""}</div><div>${esc(c.note)}</div></li>`).join("")}</ul>` : `<div class="empty">No changelog.</div>`}</div>
    <div><div class="subhead">Global events</div>${GLOBAL_EVENTS.length ? `<ul class="timeline-list">${GLOBAL_EVENTS.slice().reverse().map(e => `<li><div class="when">${esc(e.date)} · ${esc(e.label)}</div><div>${esc(e.text)}</div></li>`).join("")}</ul>` : `<div class="empty">No annotations.</div>`}
    <p class="hint">${S.showEvents ? `${evs.length} events are drawn on the time charts as ◆ markers.` : "Event markers are switched off."}</p></div></div>`;

  const cols = ["auc", "ks", "iv", "recall_at_1pct_fpr", "value_recall_at_1pct_fpr", "psi", "missing_rate", "mean", "volume"].filter(c => (tl[c] || []).some(isNum));
  const qcols = hasQ ? Object.keys(tl.quantiles) : [];
  const table = per.length ? `<div class="ctl-row"><button class="btn small" id="copyTable">${ICON.copy}Copy as TSV</button></div><div class="tablewrap"><table class="data" id="dataTable"><thead><tr><th>Period</th>${cols.map(c => `<th>${esc(METRIC[c] ? METRIC[c].short || METRIC[c].label : c)}</th>`).join("")}${qcols.map(q => `<th>${esc(q)}</th>`).join("")}</tr></thead><tbody>${per.map((p, i) => `<tr><td>${esc(p)}${i < NBASE ? ' <span class="hint">base</span>' : ""}${i > MATURE_IDX ? ' <span class="hint">immature</span>' : ""}</td>${cols.map(c => `<td class="num">${(METRIC[c] ? METRIC[c].fmt : c === "volume" ? fmt.int : fmt.num)(tl[c][i])}</td>`).join("")}${qcols.map(q => `<td class="num">${fmt.num(tl.quantiles[q][i])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>` : `<div class="empty">No timeline.</div>`;

  const body = { overview, model, definition, metadata, performance, stability, segments, distribution, drift, missing, correlations, rules: rulesBody, changelog, table };
  const visibleSecs = SECTIONS.filter(s => !S.hidden.has(s.id));
  const sopts = segOptions(f);
  $("#main").innerHTML = `<div class="detail">
    <div class="dhead">
      <div style="min-width:0">
        <h2>${esc(f.name)}</h2>
        <div class="meta">${statusBadge(f.status)}<span class="badge">${esc(f.group)}</span><span class="badge">${esc(f.dtype)}</span>${f.entity ? `<span class="badge">entity: ${esc(f.entity)}</span>` : ""}${f.owner ? `<span class="badge">owner: ${esc(f.owner)}</span>` : ""}
          <span class="tags">${(f.tags || []).map(t => `<button class="tag" data-tag="${esc(t)}" title="Filter by tag">#${esc(t)}</button>`).join("")}</span></div>
      </div>
      <div class="actions">
        <button class="btn" id="pinBtn" aria-pressed="${S.pinned.includes(f.name)}">${ICON.pin}${S.pinned.includes(f.name) ? "Pinned" : "Pin to compare"}</button>
        <button class="btn" id="linkBtn" title="Copy link to this feature">${ICON.link}Link</button>
        <button class="btn" id="prevBtn" ${idx <= 0 ? "disabled" : ""} title="Previous (k)">${ICON.prev}</button>
        <button class="btn" id="nextBtn" ${idx < 0 || idx >= list.length - 1 ? "disabled" : ""} title="Next (j)">${ICON.next}</button>
      </div>
    </div>
    <p class="desc">${esc(f.description || "")}</p>
    ${f.rationale ? `<p class="rationale"><b style="color:var(--ink)">Why it matters:</b> ${esc(f.rationale)}</p>` : ""}
    ${alerts.length ? `<div class="alerts">${alerts.map(a => `<div class="alert ${a.level}">${ICON[a.level] || ""}<div><b>${a.level === "critical" ? "Critical" : a.level === "warning" ? "Warning" : "Note"}</b>${esc(a.message)}</div>${a.sec ? `<button class="go" data-jump="${a.sec}"${a.seg ? ` data-seg="${esc(a.seg)}"` : ""}${a.metric && METRIC[a.metric] && METRIC[a.metric].labels ? ` data-metric="${a.metric}"` : ""}>View chart →</button>` : ""}</div>`).join("")}</div>` : ""}
    ${f.muted && f.mutedAlerts ? `<p class="hint">${f.mutedAlerts} alert${f.mutedAlerts > 1 ? "s" : ""} muted for this feature — see Alert rules.</p>` : ""}
    <nav class="section-bar" aria-label="Sections">${visibleSecs.map(s => `<button class="jump" data-jump="${s.id}">${esc(s.title)}</button>`).join("")}
      <span class="spacer"></span>
      ${sopts.length ? `<label class="ctl-label">Segment <select class="seg-select" id="segSelect"><option value="">All transactions</option>${sopts.map(o2 => `<option value="${esc(o2)}" ${o2 === S.segment ? "selected" : ""}>${esc(o2.replace("=", " = "))}</option>`).join("")}</select></label>` : ""}
      <label class="toggle" title="Draw changelog entries and global events on time charts"><input type="checkbox" id="evToggle" ${S.showEvents ? "checked" : ""}> Events</label>
      <button class="btn ghost small" id="expandAll">Expand all</button><button class="btn ghost small" id="collapseAll">Collapse all</button></nav>
    ${SECTIONS.map(s => card(s.id, s.title, s.sub, body[s.id])).join("")}
    ${S.hidden.size ? `<p class="hint">${S.hidden.size} section${S.hidden.size > 1 ? "s" : ""} hidden — <button class="link" id="showAllSecs">show all</button></p>` : ""}
  </div>`;
  wireDetail(f, list, idx);
  drawDetailCharts(f);
}
function wireDetail(f, list, idx) {
  const main = $("#main");
  main.querySelectorAll(".card > header").forEach(h => {
    const toggle = e => {
      if (e.target.closest(".tools")) return;
      const c = h.parentElement, id = c.dataset.sec;
      c.classList.toggle("collapsed"); const col = c.classList.contains("collapsed");
      col ? S.collapsed.add(id) : S.collapsed.delete(id); h.setAttribute("aria-expanded", !col); persist();
      if (!col) drawDetailCharts(f);
    };
    h.addEventListener("click", toggle);
    h.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(e); } });
  });
  $$("[data-hide]", main).forEach(b => b.addEventListener("click", () => { S.hidden.add(b.dataset.hide); persist(); renderDetail(); buildSectionsPanel(); toast("Section hidden — restore it from Sections"); }));
  $$("[data-jump]", main).forEach(b => b.addEventListener("click", () => {
    const id = b.dataset.jump;
    if (b.dataset.metric) { S.perfMetric = b.dataset.metric; persist(); }
    if (b.dataset.seg) { S.segment = b.dataset.seg; S.segDim = b.dataset.seg.split("=")[0]; S.segMetric = "psi"; persist(); }
    if (b.dataset.metric || b.dataset.seg || S.hidden.has(id) || S.collapsed.has(id)) { S.hidden.delete(id); S.collapsed.delete(id); persist(); renderDetail(); buildSectionsPanel(); }
    const el = document.getElementById("sec-" + id); if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  }));
  $$("[data-open]", main).forEach(b => b.addEventListener("click", () => openFeature(b.dataset.open)));
  $$("[data-tag]", main).forEach(b => b.addEventListener("click", () => { $("#q").value = "tag:" + b.dataset.tag; S.q = $("#q").value; S.view = "catalog"; refresh(); }));
  $$("[data-clearseg]", main).forEach(b => b.addEventListener("click", () => { S.segment = ""; persist(); renderDetail(); }));
  $("#expandAll").onclick = () => { S.collapsed.clear(); persist(); renderDetail(); };
  $("#collapseAll").onclick = () => { SECTIONS.forEach(s => S.collapsed.add(s.id)); persist(); renderDetail(); };
  const sa = $("#showAllSecs"); if (sa) sa.onclick = () => { S.hidden.clear(); persist(); renderDetail(); buildSectionsPanel(); };
  const ss = $("#segSelect"); if (ss) ss.onchange = () => { S.segment = ss.value; if (ss.value) S.segDim = ss.value.split("=")[0]; persist(); renderDetail(); };
  $("#evToggle").onchange = e => { S.showEvents = e.target.checked; persist(); renderDetail(); };
  $("#pinBtn").onclick = () => togglePin(f.name);
  $("#linkBtn").onclick = () => copyText(location.href.split("#")[0] + "#f=" + encodeURIComponent(f.name), "Link copied");
  $("#prevBtn").onclick = () => idx > 0 && openFeature(list[idx - 1].name);
  $("#nextBtn").onclick = () => idx < list.length - 1 && openFeature(list[idx + 1].name);
  const ls = $("#langSeg");
  if (ls) ls.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.codeLang = b.dataset.lang; persist();
    $$("button", ls).forEach(x => x.setAttribute("aria-pressed", x === b)); $("#codeBlock").innerHTML = `<code>${highlight(f.code[S.codeLang], S.codeLang === "sql" ? "sql" : "python")}</code>`; });
  const cc = $("#copyCode"); if (cc) cc.onclick = () => { const l = $("#langSeg [aria-pressed=true]").dataset.lang; copyText(f.code[l], `Copied ${l.toUpperCase()}`); };
  const segClick = (sel, key, redraw) => { const el = $(sel); if (el) el.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S[key] = b.dataset.m || b.dataset.d; persist(); $$("button", el).forEach(x => x.setAttribute("aria-pressed", x === b)); redraw(); }); };
  segClick("#perfSeg", "perfMetric", () => { $("#perfHelp").textContent = METRIC[S.perfMetric].help; drawPerf(f); });
  segClick("#distSeg", "distMode", () => drawDist(f));
  segClick("#segDimSeg", "segDim", () => drawSeg(f));
  segClick("#segMetSeg", "segMetric", () => drawSeg(f));
  ["logY", "showMean"].forEach(id => { const el = $("#" + id); if (el) el.onchange = () => { store.set(id, el.checked); drawDrift(f); }; });
  const sl = $("#shapLog"); if (sl) sl.onchange = () => { S.shapLog = sl.checked; persist(); drawModel(f); };
  const ct = $("#copyTable"); if (ct) ct.onclick = () => copyText($$("#dataTable tr").map(r => $$("th,td", r).map(c => c.textContent.replace(/ (base|immature)$/, "")).join("\t")).join("\n"), "Table copied");
  const ol = $("#openLineage"); if (ol) ol.onclick = () => { S.lgSel = "f:" + f.name; setView("lineage"); };
  const cr = $("#cmpCorr"); if (cr) cr.onclick = () => compareSet([f.name, f.correlations[0].feature]);
  const gr = $("#goRed"); if (gr) gr.onclick = () => setView("redundancy");
  const mu = $("#muteLocal"); if (mu) mu.onchange = () => { mu.checked ? MUTED.add(f.name) : MUTED.delete(f.name); store.set("muted", [...MUTED]); computeAlerts(f); refresh(); };
  const co = $("#copyOverride"); if (co) co.onclick = () => { const o = {}; o[f.name] = Object.assign({ disabled: [], note: "" }, f.alert_overrides || {}); if (MUTED.has(f.name)) o[f.name].muted = true; copyText(JSON.stringify({ alert_overrides: o[f.name] }, null, 2), "Override JSON copied — paste into this feature's entry"); };
}
function shown(id) { const el = document.getElementById("sec-" + id); return el && !el.classList.contains("collapsed"); }
function drawDetailCharts(f) { drawModel(f); drawMiniLineage(f); drawPerf(f); drawPsi(f); drawSeg(f); drawDist(f); drawDrift(f); drawMiss(f); }
function withSegment(f, metric, overall, color) {
  const sd = segData(f, S.segment), sv = sd && sd[metric];
  if (!sv) return { series: [{ name: METRIC[metric] ? METRIC[metric].label : metric, values: overall, color }], legend: [] };
  const name = S.segment.replace("=", " = ");
  return { series: [{ name, values: sv, color }, { name: "All transactions", values: overall, color: "var(--ink-2)", dash: true, width: 1.5 }],
    legend: [{ kind: "line", label: name, color }, { kind: "line", dash: true, label: "All transactions", color: "var(--ink-2)" }] };
}
function chartLegendExtras(immature) {
  const out = [];
  if (immature && MAT) out.push({ kind: "hatch", label: "labels maturing" });
  if (S.showEvents) out.push({ kind: "event", label: "event (hover)" });
  return out;
}
function drawPerf(f) {
  const el = $("#chPerf"); if (!el || !shown("performance")) return;
  const tl = f.timeline, m = S.perfMetric, vals = tl[m] || [];
  const { series, legend } = withSegment(f, m, vals, "var(--s1)");
  const base = f.kpis[m + "_base"], ev = eventsFor(f);
  if (S.segment && segData(f, S.segment) && !segData(f, S.segment)[m]) legend.push({ label: `no ${METRIC[m].short || METRIC[m].label} by segment — showing overall`, color: "transparent" });
  const host = chartHost(el, legend.concat(chartLegendExtras(true)));
  lineChart(host, {
    x: tl.periods, series, yFmt: METRIC[m].fmt, shadeBaseline: true, immature: true, events: ev,
    refs: isNum(base) && series.length === 1 ? [{ y: base, label: "baseline avg " + METRIC[m].fmt(base) }] : [], aria: METRIC[m].label + " over time",
    tipExtra: i => isNum(base) && isNum(vals[i]) ? [{ label: "overall vs baseline", value: signed(vals[i] - base, 3, METRIC[m].pp) }] : []
  });
}
function drawPsi(f) {
  const el = $("#chPsi"); if (!el || !shown("stability")) return;
  const tl = f.timeline, v = tl.psi || [], r = rulesFor(f).psi || { warning: 0.1, critical: 0.25 };
  const { series, legend } = withSegment(f, "psi", v, "var(--s2)");
  const mx = Math.max(...series.map(s => Math.max(...s.values.filter(isNum), 0)));
  const host = chartHost(el, legend.concat(chartLegendExtras(false)));
  lineChart(host, {
    x: tl.periods, series, yFmt: fmt.auc, yMin: 0, yMax: Math.max(r.critical * 1.2, mx * 1.1), shadeBaseline: true, events: eventsFor(f),
    refs: [{ y: r.warning, label: r.warning + " warning" }, { y: r.critical, label: r.critical + " critical" }], aria: "PSI over time",
    tipExtra: i => [{ label: "status", value: !isNum(series[0].values[i]) ? "—" : series[0].values[i] >= r.critical ? "critical shift" : series[0].values[i] >= r.warning ? "moderate shift" : "stable" }]
  });
}
function drawMiss(f) {
  const el = $("#chMiss"); if (!el || !shown("missing")) return;
  const tl = f.timeline, v = tl.missing_rate, base = f.kpis.missing_rate_base;
  const { series, legend } = withSegment(f, "missing_rate", v, "var(--s3)");
  const mx = Math.max(...series.map(s => Math.max(...s.values.filter(isNum), 0)));
  const host = chartHost(el, legend.concat(chartLegendExtras(false)));
  lineChart(host, { x: tl.periods, series, yFmt: fmt.pct, yMin: 0, shadeBaseline: true, yMax: Math.max(0.05, mx) * 1.15, aria: "missing rate over time", events: eventsFor(f),
    refs: isNum(base) && base > 0 && series.length === 1 ? [{ y: base, label: "baseline " + fmt.pct(base) }] : [],
    tipExtra: i => (tl.volume && isNum(tl.volume[i])) ? [{ label: "txn volume", value: fmt.int(tl.volume[i]) }, { label: "null txns (overall)", value: isNum(v[i]) ? fmt.int(tl.volume[i] * v[i]) : "—" }] : [] });
}
function drawSeg(f) {
  const el = $("#chSeg"); if (!el || !shown("segments")) return;
  const dims = Object.keys(f.segments || {}); if (!dims.length) return;
  const dim = dims.includes(S.segDim) ? S.segDim : dims[0], m = S.segMetric, tl = f.timeline;
  const vals = Object.keys(f.segments[dim]);
  const rows = [{ key: "", label: "All transactions" }].concat(vals.map(v => { const sh = (f.segments[dim][v].share || []).slice(-1)[0]; return { key: dim + "=" + v, label: v + (isNum(sh) ? ` · ${(sh * 100).toFixed(sh < 0.1 ? 1 : 0)}%` : "") }; }));
  const series = r => r === 0 ? (m === "share" ? tl.periods.map(() => 1) : tl[m] || []) : (f.segments[dim][vals[r - 1]][m] || []);
  const isDelta = m === "auc" || m === "recall_at_1pct_fpr";
  const base = r => mean(series(r).slice(0, NBASE));
  const value = (r, c) => { const v = series(r)[c]; if (!isNum(v)) return null; return isDelta ? (isNum(base(r)) ? v - base(r) : null) : v; };
  const allv = []; rows.forEach((_, r) => tl.periods.forEach((_, c) => { const v = value(r, c); if (isNum(v)) allv.push(Math.abs(v)); }));
  const rr = rulesFor(f);
  const hi = m === "psi" ? Math.max(rr.segment_psi ? rr.segment_psi.critical : 0.35, 0.25) : isDelta ? Math.max(0.05, ...allv) : Math.max(0.01, ...allv);
  const color = v => isDelta ? divColor(v / hi) : seqColor(Math.min(1, v / hi));
  const vf = v => isDelta ? (m === "auc" ? signed(v, 2) : signed(v, 0, true)) : m === "psi" ? v.toFixed(2) : fmt.pct(v);
  const lo = isDelta ? "−" + (m === "auc" ? hi.toFixed(2) : (hi * 100).toFixed(0) + "pp") : "0";
  const hiL = isDelta ? "+" + (m === "auc" ? hi.toFixed(2) : (hi * 100).toFixed(0) + "pp") : m === "psi" ? "≥ " + hi.toFixed(2) : fmt.pct(hi);
  $("#segLegend").innerHTML = isDelta ? gradLegend(lo, hiL, cssVar("--div-neg"), cssVar("--div-pos"), cssVar("--div-mid")) + `<span class="hint" style="margin-left:6px">change vs each row's baseline</span>` : gradLegend(lo, hiL, cssVar("--seq-lo"), cssVar("--seq-hi"));
  heatmap(el, {
    rows, cols: tl.periods, value, color, fmt: vf, showValues: true, labelW: 140, colLabels: shortPeriod, aria: "segment heatmap",
    selectedRow: Math.max(0, rows.findIndex(x => x.key === S.segment)),
    tip: (r, c) => {
      const sd = r === 0 ? null : f.segments[dim][vals[r - 1]], raw = series(r)[c];
      const out = [{ label: METRIC[m] ? METRIC[m].label : m, value: isNum(raw) ? (METRIC[m] ? METRIC[m].fmt(raw) : raw) : "—" }];
      if (isDelta && isNum(value(r, c))) out.push({ label: "vs row baseline", value: vf(value(r, c)) });
      if (sd && sd.share) out.push({ label: "volume share", value: fmt.pct(sd.share[c]) });
      if (c > MATURE_IDX && METRIC[m] && METRIC[m].labels) out.push({ note: "Labels still maturing for this period." });
      eventsFor(f).filter(e => e.i === c).forEach(e => out.push({ note: `<b>◆ ${esc(e.label)}</b> ${esc(e.text)}`, html: true }));
      return tipRows(rows[r].label + " · " + tl.periods[c], out);
    },
    onRow: r => { S.segment = rows[r].key; persist(); renderDetail(); toast(r ? `Overlaying ${rows[r].label} on time charts` : "Showing all transactions"); }
  });
}
function drawModel(f) {
  const mb = f.model_behavior || {};
  const sh = $("#chShap");
  if (sh && shown("model")) {
    if (f.dtype === "numeric") scatterChart(sh, { points: mb.shap_points, xLog: S.shapLog });
    else { const d = f.distribution || {}; stripChart(sh, { points: mb.shap_points, cats: (d.bins || []).filter(b => b !== "Missing") }); }
  }
  const im = $("#chImp");
  if (im && shown("model") && (mb.importance_timeline || []).length) {
    lineChart(im, { x: PERIODS, series: [{ name: "mean |SHAP|", values: mb.importance_timeline, color: "var(--s1)" }], yFmt: v => isNum(v) ? v.toFixed(3) : "—", zeroBased: true, shadeBaseline: true, events: eventsFor(f),
      tipExtra: i => [{ label: "rank that period", value: RANK_TL[i] && RANK_TL[i][f.name] ? "#" + RANK_TL[i][f.name] : "—" }] });
  }
}
function drawMiniLineage(f) {
  const el = $("#miniLineage"); if (!el || !shown("metadata")) return;
  const id = "f:" + f.name;
  if (!LG.nodes.has(id)) { el.innerHTML = `<div class="empty" style="padding:10px">No lineage.</div>`; return; }
  const ids = new Set([id, ...closure(id, "up"), ...closure(id, "down")]);
  renderLineage(el, { ids, focus: id, onSelect: nid => { const n = LG.nodes.get(nid); if (n.f && n.f.name !== f.name) openFeature(n.f.name); else if (n.kind === "table") { S.lgSel = nid; setView("lineage"); } } });
}
function drawDist(f) {
  const host = $("#distCharts"); if (!host || !shown("distribution")) return;
  const d = f.distribution, mode = S.distMode, base = MODEL.base_fraud_rate;
  const lastP = (f.timeline && f.timeline.periods || []).slice(-1)[0] || "latest", frP = d.fraud_rate_period || lastP;
  host.innerHTML = (mode !== "fraud" ? `<div><div class="subhead">Share of transactions per bin</div>${legendHTML([{ label: "Baseline", color: "var(--s1)" }, { label: "Latest (" + lastP + ")", color: "var(--s2)" }])}<div class="chart" id="chDist1"></div></div>` : "") +
    (mode !== "shares" ? `<div style="${mode === "both" ? "margin-top:16px" : ""}"><div class="subhead">Fraud rate per bin (${esc(frP)}${d.fraud_rate_period && d.fraud_rate_period !== lastP ? ", last period with mature labels" : ""})</div>${legendHTML([{ label: "Fraud rate", color: "var(--s1)" }].concat(isNum(base) ? [{ kind: "line", dash: true, label: "Portfolio average", color: "var(--ink-2)" }] : []))}<div class="chart" id="chDist2"></div></div>` : "");
  const tipPsi = i => { const b = d.baseline[i], c = d.current[i]; return isNum(b) && isNum(c) ? [{ label: "PSI contribution", value: ((c - b) * Math.log((c + 1e-4) / (b + 1e-4))).toFixed(4) }] : []; };
  if (mode !== "fraud") barChart($("#chDist1"), { cats: d.bins, catTitle: "Bin", series: [{ name: "Baseline", values: d.baseline, color: "var(--s1)" }, { name: "Latest", values: d.current, color: "var(--s2)" }], yFmt: fmt.pct, tipExtra: tipPsi, aria: "distribution baseline vs latest" });
  if (mode !== "shares") barChart($("#chDist2"), { cats: d.bins, catTitle: "Bin", series: [{ name: "Fraud rate", values: d.fraud_rate || [], color: "var(--s1)" }], yFmt: fmt.rate,
    refs: isNum(base) ? [{ y: base, label: "avg " + fmt.rate(base) }] : [], aria: "fraud rate per bin",
    tipExtra: i => isNum(base) && isNum((d.fraud_rate || [])[i]) ? [{ label: "lift vs avg", value: (d.fraud_rate[i] / base).toFixed(2) + "×" }, { label: "share of txns", value: fmt.pct(d.current[i]) }] : [] });
}
function drawDrift(f) {
  const el = $("#chDrift"); if (!el || !shown("drift")) return;
  const tl = f.timeline, ev = eventsFor(f);
  if (tl.quantiles && Object.keys(tl.quantiles).length) {
    const q = tl.quantiles, log = $("#logY") && $("#logY").checked, showMean = $("#showMean") && $("#showMean").checked;
    const tf = log ? (a => (a || []).map(v => isNum(v) ? Math.log10(Math.max(v, 0) + 1) : null)) : (a => a || []);
    const yFmt = log ? (v => fmt.num(Math.pow(10, v) - 1)) : fmt.num;
    const series = [{ name: "median (p50)", values: tf(q.p50), color: "var(--s1)" }];
    if (showMean && (tl.mean || []).some(isNum)) series.push({ name: "mean", values: tf(tl.mean), color: "var(--s2)", dash: true });
    const host = chartHost(el, [{ kind: "line", label: "Median", color: "var(--s1)" }, { label: "p25–p75", color: "var(--s1-band2)" }, { label: "p05–p95", color: "var(--s1-band)" }]
      .concat(series.length > 1 ? [{ kind: "line", dash: true, label: "Mean", color: "var(--s2)" }] : []).concat(chartLegendExtras(false)));
    const nonNeg = !log && (q.p05 || []).every(v => !isNum(v) || v >= 0);
    lineChart(host, { x: tl.periods, height: 240, series, yFmt, shadeBaseline: true, aria: "quantiles over time", zeroBased: nonNeg, events: ev,
      bands: [{ lo: tf(q.p05), hi: tf(q.p95), fill: "var(--s1-band)", name: "p05–p95" }, { lo: tf(q.p25), hi: tf(q.p75), fill: "var(--s1-band2)", name: "p25–p75" }] });
  } else if (tl.category_shares) {
    const cats = Object.keys(tl.category_shares);
    const avg = c => mean(tl.category_shares[c]);
    cats.sort((a, b) => avg(b) - avg(a));
    const top = cats.slice(0, cats.length > 5 ? 4 : 5), rest = cats.slice(top.length);
    const series = top.map((c, i) => ({ name: c, values: tl.category_shares[c], color: i < 4 ? seriesColor(i) : "var(--muted)" }));
    if (rest.length) series.push({ name: `other (${rest.length})`, values: tl.periods.map((_, i) => rest.reduce((a, c) => a + (tl.category_shares[c][i] || 0), 0)), color: "var(--muted)" });
    const host = chartHost(el, series.map(s => ({ label: s.name, color: s.color })));
    stackChart(host, { x: tl.periods, series, height: 240, events: ev });
  }
}

/* ---- compare */
function renderCompare() {
  const fs = S.pinned.map(n => BYNAME[n]).filter(Boolean);
  if (!fs.length) {
    $("#main").innerHTML = `<div class="detail"><h2 style="font-size:18px;margin:0 0 8px">Compare</h2><p class="hint">Pin up to 4 features with the ${ICON.pin.replace('class="icon"', 'class="icon" style="vertical-align:-3px"')} button in the list or catalog (or press <kbd>x</kbd> on a feature) to overlay their metrics here. Redundancy groups can be sent here in one click.</p></div>`;
    return;
  }
  const metrics = [...LABEL_METRICS, "psi", "missing_rate"];
  if (!metrics.includes(S.cmpMetric)) S.cmpMetric = "auc";
  const pairs = []; for (let i = 0; i < fs.length; i++) for (let j = i + 1; j < fs.length; j++) { const r = corr(fs[i].name, fs[j].name); if (isNum(r)) pairs.push([fs[i].name, fs[j].name, r]); }
  $("#main").innerHTML = `<div class="detail">
    <div class="dhead"><div><h2 style="font-size:18px">Compare ${fs.length} feature${fs.length > 1 ? "s" : ""}</h2></div>
      <div class="actions"><button class="btn" id="clearPins">Unpin all</button></div></div>
    <div class="ctl-row" style="margin-top:12px"><div class="seg" id="cmpSeg">${metrics.map(m => `<button data-m="${m}" aria-pressed="${S.cmpMetric === m}">${esc(METRIC[m].short || METRIC[m].label)}</button>`).join("")}</div><span class="hint" id="cmpHelp">${esc(METRIC[S.cmpMetric].help)}</span></div>
    <div class="card"><div class="body" style="padding-top:12px"><div id="chCmp"></div></div></div>
    <div class="card"><div class="body" style="padding-top:12px"><div class="tablewrap"><table class="data"><thead><tr><th>Feature</th><th>Rank</th><th>$ at risk</th>${metrics.map(m => `<th>${esc(METRIC[m].short || METRIC[m].label)}</th><th>Δ base</th>`).join("")}<th></th></tr></thead><tbody>
      ${fs.map((f, i) => `<tr><td><span class="legend" style="display:inline"><span class="ln" style="border-color:${seriesColor(i)}"></span></span><button class="link" data-open="${esc(f.name)}">${esc(f.name)}</button></td><td class="num">${rank(f) < 1e9 ? rank(f) : "—"}</td><td class="num">${fmt.usd(f.kpis.usd_at_risk)}</td>
        ${metrics.map(m => { const c = f.kpis[m], b = f.kpis[m + "_base"]; return `<td class="num">${METRIC[m].fmt(c)}</td><td class="num">${isNum(c) && isNum(b) ? signed(c - b, 3, METRIC[m].pp) : "—"}</td>`; }).join("")}
        <td><button class="pin" data-pin="${esc(f.name)}" aria-pressed="true" title="Unpin">${ICON.pin}</button></td></tr>`).join("")}
    </tbody></table></div>
    ${pairs.length ? `<p class="hint" style="margin-top:10px">Pairwise correlation: ${pairs.map(([a, b, r]) => `<code>${esc(a)}</code> × <code>${esc(b)}</code> <b>${r >= 0 ? "+" : "−"}${Math.abs(r).toFixed(2)}</b>`).join(" · ")}</p>` : ""}
    </div></div>
    <div class="subhead" style="margin-top:6px">Descriptions</div>
    ${fs.map(f => `<p style="margin:4px 0"><code>${esc(f.name)}</code> — <span style="color:var(--ink-2)">${esc(f.description || "")}</span></p>`).join("")}
  </div>`;
  const draw = () => {
    const m = S.cmpMetric, per = (fs[0].timeline || {}).periods || [];
    const host = chartHost($("#chCmp"), fs.map((f, i) => ({ kind: "line", label: f.name, color: seriesColor(i) })).concat(chartLegendExtras(METRIC[m].labels)));
    lineChart(host, { x: per, height: 260, shadeBaseline: true, immature: METRIC[m].labels, events: eventsFor(null), yFmt: METRIC[m].fmt, yMin: m === "psi" || m === "missing_rate" ? 0 : undefined,
      series: fs.map((f, i) => ({ name: f.name, values: (f.timeline || {})[m] || [], color: seriesColor(i) })),
      refs: m === "psi" ? [{ y: RULES.psi.warning, label: String(RULES.psi.warning) }, { y: RULES.psi.critical, label: String(RULES.psi.critical) }] : [] });
  };
  draw();
  $("#cmpSeg").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.cmpMetric = b.dataset.m; persist();
    $$("#cmpSeg button").forEach(x => x.setAttribute("aria-pressed", x === b)); $("#cmpHelp").textContent = METRIC[S.cmpMetric].help; draw(); });
  $("#clearPins").onclick = () => { S.pinned = []; persist(); refresh(); };
  $$("[data-open]", $("#main")).forEach(b => b.addEventListener("click", () => openFeature(b.dataset.open)));
  $$("[data-pin]", $("#main")).forEach(b => b.addEventListener("click", () => togglePin(b.dataset.pin)));
}

/* ---- lineage view */
function renderLineageView() {
  const vis = new Set(visible().map(f => "f:" + f.name));
  const ids = new Set();
  LG.nodes.forEach((n, id) => {
    if (n.kind === "feature" && n.f && !vis.has(id)) return;
    if (n.kind === "feature" && n.f && !S.lgDeprecated && n.f.status === "deprecated") return;
    ids.add(id);
  });
  // drop tables / external nodes left without a visible child
  [...ids].forEach(id => { const n = LG.nodes.get(id); if ((n.kind === "table" || !n.f) && ![...n.children].some(c => ids.has(c))) ids.delete(id); });
  const sel = S.lgSel && LG.nodes.has(S.lgSel) ? LG.nodes.get(S.lgSel) : null;
  let panel = `<div class="hint">Hover a node to trace everything upstream and downstream of it. Click a source table to see its blast radius; click a feature to see its sources.</div>`;
  if (sel) {
    const down = [...closure(sel.id, "down")].map(id => LG.nodes.get(id)).filter(n => n.f).map(n => n.f).sort((a, b) => rank(a) - rank(b));
    const up = [...closure(sel.id, "up")].map(id => LG.nodes.get(id));
    const shapTot = FEATS.reduce((a, f) => a + ((f.importance || {}).shap_mean_abs || 0), 0) || 1;
    const shapDown = down.reduce((a, f) => a + ((f.importance || {}).shap_mean_abs || 0), 0);
    const usd = down.reduce((a, f) => a + (f.kpis.usd_at_risk || 0), 0);
    panel = `<h4>${esc(sel.label)}</h4><div class="hint">${sel.kind === "table" ? "source table" : "feature"}</div>
      ${sel.f ? `<p style="margin:8px 0"><button class="btn small" data-open="${esc(sel.f.name)}">Open feature</button></p>` : ""}
      ${down.length ? `<div class="subhead" style="margin-top:10px">Blast radius: ${down.length} feature${down.length > 1 ? "s" : ""}</div>
        <p class="help">${(shapDown / shapTot * 100).toFixed(0)}% of total model importance · ≈${fmt.usd(usd)}/mo of fraud catch from drop-column tests (not additive) · ${down.filter(f => maxSev(f) >= 2).length} with alerts.</p>
        <ul>${down.map(f => `<li><button class="link" data-open="${esc(f.name)}">${esc(f.name)}</button><span class="hint">#${rank(f)}${maxSev(f) >= 2 ? " · ⚠" : ""}</span></li>`).join("")}</ul>` : ""}
      ${up.length ? `<div class="subhead" style="margin-top:10px">Depends on</div><ul>${up.map(n => `<li>${n.f ? `<button class="link" data-open="${esc(n.f.name)}">${esc(n.label)}</button>` : `<code>${esc(n.label)}</code>`}<span class="hint">${n.kind}</span></li>`).join("")}</ul>` : ""}`;
  }
  const nT = [...ids].filter(id => LG.nodes.get(id).kind === "table").length, nF = ids.size - nT;
  $("#main").innerHTML = `<div class="catalog">
    <div class="ctl-row"><h2 style="font-size:18px;margin:0 12px 0 0">Lineage</h2>
      <div class="search" style="width:260px"><svg class="icon" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg><input id="lgQ" type="search" placeholder="Highlight table or feature…" value="${esc(S.lgQuery)}"></div>
      <label class="toggle"><input type="checkbox" id="lgDep" ${S.lgDeprecated ? "checked" : ""}> Show deprecated</label>
      <span class="hint">${nT} source tables → ${nF} features · respects sidebar filters · node border = alert level</span></div>
    ${legendHTML([{ label: "source table", color: "var(--surface-3)" }, { label: "feature", color: "var(--surface)" }, { label: "critical alert", color: "var(--critical)" }, { label: "warning", color: "var(--warning)" }])}
    <div class="lg-layout"><div class="lg-wrap" id="lgMain"></div><aside class="side-panel">${panel}</aside></div></div>`;
  renderLineage($("#lgMain"), { ids, selected: S.lgSel, query: S.lgQuery, onSelect: id => { S.lgSel = id; renderLineageView(); } });
  $("#lgQ").addEventListener("input", e => { S.lgQuery = e.target.value; renderLineage($("#lgMain"), { ids, selected: S.lgSel, query: S.lgQuery, onSelect: id => { S.lgSel = id; renderLineageView(); } }); });
  $("#lgDep").onchange = e => { S.lgDeprecated = e.target.checked; persist(); renderLineageView(); };
  $$("[data-open]", $("#main")).forEach(b => b.addEventListener("click", () => openFeature(b.dataset.open)));
  if (S.lgSel) { const n = $(`.lg-node[data-id="${CSS.escape(S.lgSel)}"]`); if (n) n.scrollIntoView({ block: "center", inline: "center" }); }
}

/* ---- redundancy view */
function renderRedundancy() {
  if (!CM) { $("#main").innerHTML = `<div class="detail"><div class="empty">No correlation matrix in the input (add top-level "correlations": {"features": [...], "matrix": [[...]]}).</div></div>`; return; }
  const list = visible().filter(f => CIDX[f.name] != null);
  let names = list.map(f => f.name);
  if (S.redOrder === "cluster") names = clusterOrder(names);
  else if (S.redOrder === "group") names.sort((a, b) => BYNAME[a].group.localeCompare(BYNAME[b].group) || rank(BYNAME[a]) - rank(BYNAME[b]));
  else names.sort((a, b) => rank(BYNAME[a]) - rank(BYNAME[b]));
  const thr = S.redThr, groups = redundancyGroups(list, thr);
  const gHtml = groups.length ? groups.map((g, gi) => {
    const keep = g[0];
    let mxr = 0; for (let i = 0; i < g.length; i++) for (let j = i + 1; j < g.length; j++) mxr = Math.max(mxr, Math.abs(corr(g[i], g[j]) || 0));
    return `<div class="rgroup"><header><span><b style="color:var(--ink)">Group ${gi + 1}</b> · ${g.length} features · max |r| ${mxr.toFixed(2)}</span><button class="btn small" data-cmp="${gi}">Compare</button></header>
      ${g.map(n => { const f = BYNAME[n], r = n === keep ? null : corr(n, keep);
        return `<div class="mem"><div><button class="link" data-open="${esc(n)}">${esc(n)}</button> ${n === keep ? '<span class="keep">keep</span>' : '<span class="review">review</span>'}</div>
          <span class="hint num">#${rank(f)} · ${fmt.usd(f.kpis.usd_at_risk)}</span><span class="num" style="min-width:48px;text-align:right">${r == null ? "" : (r >= 0 ? "+" : "−") + Math.abs(r).toFixed(2)}</span></div>`; }).join("")}
    </div>`; }).join("") : `<div class="empty">No pairs at |r| ≥ ${thr.toFixed(2)} among the ${list.length} features shown.</div>`;
  $("#main").innerHTML = `<div class="catalog">
    <div class="ctl-row"><h2 style="font-size:18px;margin:0 12px 0 0">Correlation &amp; redundancy</h2>
      <label class="ctl-label">Threshold |r| ≥ <input type="range" id="thr" min="0.5" max="0.95" step="0.01" value="${thr}"><b class="num" id="thrV">${thr.toFixed(2)}</b></label>
      <div class="seg" id="ordSeg">${[["cluster", "Clustered"], ["group", "By group"], ["rank", "By rank"]].map(([k, l]) => `<button data-o="${k}" aria-pressed="${S.redOrder === k}">${l}</button>`).join("")}</div>
      ${gradLegend("−1", "+1", cssVar("--div-neg"), cssVar("--div-pos"), cssVar("--div-mid"))}
      <span class="hint">${esc((DATA.correlations || {}).method || "")} · outlined cells meet the threshold · click a cell to compare the pair</span></div>
    <div class="red-layout"><div class="lg-wrap" style="padding:8px" id="corrHm"></div><aside><div class="subhead">Redundancy groups (${groups.length})</div>
      <p class="help">Connected components of |r| ≥ threshold. “Keep” = highest mean |SHAP|; review the others — check their drop-column $ at risk before removing.</p>${gHtml}</aside></div></div>`;
  const maxCell = Math.max(12, Math.min(26, Math.floor(($("#corrHm").clientWidth - 230) / Math.max(1, names.length))));
  heatmap($("#corrHm"), {
    rows: names.map(n => ({ key: n, label: n })), cols: names, square: true, maxCell, labelW: 220, monoRows: true, aria: "correlation matrix",
    value: (r, c) => corr(names[r], names[c]), color: v => divColor(v), fmt: v => v.toFixed(2),
    outline: (r, c) => r !== c && Math.abs(corr(names[r], names[c]) || 0) >= thr,
    tip: (r, c) => tipRows(`${names[r]} × ${names[c]}`, [{ label: "r", value: (corr(names[r], names[c]) || 0).toFixed(3) }, { label: "groups", value: BYNAME[names[r]].group + " / " + BYNAME[names[c]].group }]),
    onCell: (r, c) => { if (r !== c) compareSet([names[r], names[c]]); else openFeature(names[r]); },
    onRow: r => openFeature(names[r])
  });
  const t = $("#thr");
  t.addEventListener("input", () => { $("#thrV").textContent = (+t.value).toFixed(2); });
  t.addEventListener("change", () => { S.redThr = +t.value; persist(); renderRedundancy(); });
  $("#ordSeg").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.redOrder = b.dataset.o; persist(); renderRedundancy(); });
  $$("[data-open]", $("#main")).forEach(b => b.addEventListener("click", () => openFeature(b.dataset.open)));
  $$("[data-cmp]", $("#main")).forEach(b => b.addEventListener("click", () => compareSet(groups[+b.dataset.cmp])));
}

/* ---------------------------------------------------------------- rules menu */
function buildRulesPanel() {
  const fields = ["warning", "critical", "lookback", "min_share"];
  const edited = (k, f) => RULES_FILE[k] && RULES[k][f] !== RULES_FILE[k][f];
  $("#rulesPanel").innerHTML = `<h4>Alert rules (all features)</h4>
    <table class="rules-tbl"><thead><tr><th>Rule</th>${fields.map(f => `<th>${f.replace("_", " ")}</th>`).join("")}</tr></thead><tbody>
    ${Object.keys(RULE_META).filter(k => RULES[k]).map(k => `<tr><td title="${esc(k)}">${esc(RULE_META[k].label)}</td>${fields.map(f => RULE_META[k].fields.includes(f) ? `<td><input type="number" step="${f === "lookback" ? 1 : 0.01}" min="0" data-k="${k}" data-f="${f}" value="${RULES[k][f] != null ? RULES[k][f] : ""}" class="${edited(k, f) ? "edited" : ""}"></td>` : "<td></td>").join("")}</tr>`).join("")}
    </tbody></table>
    <p class="hint" style="margin:8px 6px 0">Changes apply instantly and are saved in this browser. Per-feature overrides from the data file still win. Missing / recall thresholds are absolute (0.05 = 5pp).</p>
    <div class="foot"><button class="btn small" id="rulesReset">Reset to file</button><button class="btn small" id="rulesCopy">${ICON.copy}Copy as JSON</button></div>`;
  $$("#rulesPanel input").forEach(inp => inp.addEventListener("change", () => {
    const v = inp.value === "" ? null : +inp.value; RULES = deepMerge(RULES, { [inp.dataset.k]: { [inp.dataset.f]: v } });
    const diff = {}; Object.keys(RULES).forEach(k => Object.keys(RULES[k]).forEach(f => { if (!RULES_FILE[k] || RULES[k][f] !== RULES_FILE[k][f]) (diff[k] = diff[k] || {})[f] = RULES[k][f]; }));
    store.set("rules", diff); recomputeAll(); buildRulesPanel(); refresh();
  }));
  $("#rulesReset").onclick = () => { RULES = clone(RULES_FILE); store.set("rules", {}); recomputeAll(); buildRulesPanel(); refresh(); toast("Rules reset to the file defaults"); };
  $("#rulesCopy").onclick = () => copyText(JSON.stringify(RULES, null, 2), "Rules JSON copied — save it and pass --rules to the generator");
}

/* ---------------------------------------------------------------- sections menu, theme, keys */
function buildSectionsPanel() {
  $("#sectionsPanel").innerHTML = `<h4>Show sections</h4>` + SECTIONS.map(s => `<label><input type="checkbox" data-s="${s.id}" ${S.hidden.has(s.id) ? "" : "checked"}> ${esc(s.title)}</label>`).join("") +
    `<div class="foot"><button class="btn small" data-preset="all">Show all</button><button class="btn small" data-preset="minimal">Essentials</button><button class="btn small" data-preset="monitor">Monitoring</button></div>
     <div class="foot"><button class="btn small" data-preset="model">Model review</button><button class="btn small ghost" data-preset="reset">Reset layout</button></div>`;
}
$("#sectionsPanel").addEventListener("change", e => {
  const c = e.target.closest("[data-s]"); if (!c) return;
  c.checked ? S.hidden.delete(c.dataset.s) : S.hidden.add(c.dataset.s); persist(); if (S.view === "detail") renderDetail();
});
$("#sectionsPanel").addEventListener("click", e => {
  const b = e.target.closest("[data-preset]"); if (!b) return;
  const p = b.dataset.preset;
  const keep = { all: SECTIONS.map(s => s.id), minimal: ["overview", "definition", "performance", "stability"],
    monitor: ["overview", "performance", "stability", "segments", "drift", "missing", "rules", "changelog"],
    model: ["overview", "model", "definition", "performance", "correlations", "metadata"] }[p];
  if (p === "reset") { S.hidden.clear(); S.collapsed = new Set(DEFAULT_COLLAPSED); }
  else S.hidden = new Set(SECTIONS.map(s => s.id).filter(id => !keep.includes(id)));
  persist(); buildSectionsPanel(); if (S.view === "detail") renderDetail();
});
function closePops(except) { $$(".pop.open").forEach(p => p !== except && p.classList.remove("open")); }
["sectionsPop", "helpPop", "rulesPop"].forEach(id => {
  const pop = document.getElementById(id);
  pop.querySelector("button").addEventListener("click", e => { e.stopPropagation(); closePops(pop); pop.classList.toggle("open"); });
  pop.querySelector(".pop-panel").addEventListener("click", e => e.stopPropagation());
});
document.addEventListener("click", () => closePops());

function applyTheme(t) { if (t) document.documentElement.setAttribute("data-theme", t); else document.documentElement.removeAttribute("data-theme"); }
applyTheme(store.get("theme", null));
function toggleTheme() {
  const cur = document.documentElement.getAttribute("data-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark"; applyTheme(next); store.set("theme", next); refresh();
}
$("#themeBtn").onclick = toggleTheme;
$("#toggleSidebar").onclick = () => { S.sidebar = !S.sidebar; persist(); refresh(); };

function step(dir) {
  const list = visible(); if (!list.length) return;
  let i = list.findIndex(f => f.name === S.current);
  i = i < 0 ? 0 : Math.max(0, Math.min(list.length - 1, i + dir));
  openFeature(list[i].name);
}
document.addEventListener("keydown", e => {
  const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
  if (e.key === "Escape") { closePops(); if (typing && e.target.id === "q") { e.target.value = ""; S.q = ""; refresh(); e.target.blur(); } return; }
  if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key;
  if (k === "/") { e.preventDefault(); if (!S.sidebar) { S.sidebar = true; persist(); refresh(); } $("#q").focus(); $("#q").select(); }
  else if (k === "j") step(1);
  else if (k === "k") step(-1);
  else if (k === "c") setView("catalog");
  else if (k === "d") setView("detail");
  else if (k === "p") setView("compare");
  else if (k === "l") setView("lineage");
  else if (k === "r") setView("redundancy");
  else if (k === "t") toggleTheme();
  else if (k === "[") { S.sidebar = !S.sidebar; persist(); refresh(); }
  else if (k === "x" && S.view === "detail" && S.current) togglePin(S.current);
  else if (k === "e" && S.view === "detail") { if (S.collapsed.size) S.collapsed.clear(); else SECTIONS.forEach(s => S.collapsed.add(s.id)); persist(); renderDetail(); }
  else if (k === "?") { e.preventDefault(); const p = $("#helpPop"); closePops(p); p.classList.toggle("open"); }
});

let rz; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => {
  if (S.view === "detail" && BYNAME[S.current]) drawDetailCharts(BYNAME[S.current]); else if (S.view !== "catalog") refresh();
}, 120); });
window.addEventListener("hashchange", () => { readHash(); refresh(); });

/* ---------------------------------------------------------------- boot */
$("#title").textContent = DATA.title || "Feature Explorer";
$("#modelInfo").textContent = [MODEL.name, MODEL.version, MODEL.target ? "target: " + MODEL.target : "", isNum(MODEL.base_fraud_rate) ? "base rate " + fmt.rate(MODEL.base_fraud_rate) : "", DATA.generated_at ? "generated " + DATA.generated_at : ""].filter(Boolean).join(" · ");
initFilters(); buildSectionsPanel(); buildRulesPanel(); readHash();
if (S.view === "detail" && !S.current && FEATS.length) S.current = visible()[0].name;
refresh();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    sys.exit(main())
