
import argparse
import sys
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd


INDICATORS_CONFIG = {
    "car_pct": {
        "name": "Capital Adequacy Ratio (%)",
        "direction": "safer_when_higher",
        "method": "thresholds",
        "thresholds": [8, 10, 12, 14],
        "weight": 0.20,
    },
    "npl_pct": {
        "name": "Non-Performing Loans (%)",
        "direction": "riskier_when_higher",
        "method": "thresholds",
        # ілюстративно: 3, 5, 8, 12
        "thresholds": [3, 5, 8, 12],
        "weight": 0.20,
    },
    "lcr_pct": {
        "name": "Liquidity Coverage Ratio (%)",
        "direction": "safer_when_higher",
        "method": "thresholds",
        "thresholds": [80, 100, 120, 140],
        "weight": 0.15,
    },
    "roa_pct": {
        "name": "Return on Assets (%)",
        "direction": "safer_when_higher",
        "method": "quantiles",
        "quantiles": [0.2, 0.4, 0.6, 0.8],
        "weight": 0.10,
    },
    "ops_incidents_per_10k": {
        "name": "Operational Incidents / 10k tx",
        "direction": "riskier_when_higher",
        "method": "quantiles",
        "quantiles": [0.2, 0.4, 0.6, 0.8],
        "weight": 0.10,
    },
    "fail_rate_pct": {
        "name": "Payment Fail Rate (%)",
        "direction": "riskier_when_higher",
        "method": "thresholds",
        "thresholds": [0.2, 0.4, 0.6, 1.0],
        "weight": 0.15,
    },
    "interbank_centrality": {
        "name": "Interbank Network Centrality (0..1)",
        "direction": "riskier_when_higher",
        "method": "thresholds",
        "thresholds": [0.2, 0.4, 0.6, 0.8],
        "weight": 0.10,
    },
}


def ordinalize_by_thresholds(x: float, t: List[float], higher_is_riskier=True) -> int:

    if pd.isna(x):
        return np.nan

    if x <= t[0]:
        cat = 1
    elif x <= t[1]:
        cat = 2
    elif x <= t[2]:
        cat = 3
    elif x <= t[3]:
        cat = 4
    else:
        cat = 5
    if not higher_is_riskier:

        cat = 6 - cat
    return cat


def ordinalize_by_quantiles(series: pd.Series, quantiles: List[float], higher_is_riskier=True) -> pd.Series:

    qs = series.quantile(quantiles).values.tolist()
    cats = []
    for x in series:
        if pd.isna(x):
            cats.append(np.nan)
            continue
        if x <= qs[0]:
            c = 1
        elif x <= qs[1]:
            c = 2
        elif x <= qs[2]:
            c = 3
        elif x <= qs[3]:
            c = 4
        else:
            c = 5
        if not higher_is_riskier:
            c = 6 - c
        cats.append(c)
    return pd.Series(cats, index=series.index, name=series.name)


def normalize_ori(ori_raw: pd.Series, min_val: Optional[float] = None, max_val: Optional[float] = None) -> pd.Series:

    if min_val is None:
        min_val = np.nanmin(ori_raw.values)
    if max_val is None:
        max_val = np.nanmax(ori_raw.values)
    if np.isclose(max_val, min_val):
        return pd.Series(0.5, index=ori_raw.index)
    return (ori_raw - min_val) / (max_val - min_val)


def classify_tertiles(score: pd.Series, labels=("Low", "Medium", "High")) -> pd.Series:

    q1 = score.quantile(1/3)
    q2 = score.quantile(2/3)
    out = []
    for x in score:
        if x <= q1:
            out.append(labels[0])
        elif x <= q2:
            out.append(labels[1])
        else:
            out.append(labels[2])
    return pd.Series(out, index=score.index, name="risk_class")


def compute_hurwicz(ordinal_levels_df: pd.DataFrame, alpha: float = 0.7) -> pd.Series:

    worst = ordinal_levels_df.max(axis=1, skipna=True)
    mean_ = ordinal_levels_df.mean(axis=1, skipna=True)
    hurwicz = alpha * worst + (1 - alpha) * mean_
    # Нормалізація до [0,1] (1..5 -> 0..1)
    hurwicz_norm = (hurwicz - 1) / 4.0
    return pd.Series(hurwicz_norm, index=ordinal_levels_df.index, name="hurwicz_risk")

def ordinal_risk_pipeline(
    df: pd.DataFrame,
    indicators_config: Dict[str, Dict],
    bank_col: str = "bank",
    hurwicz_alpha: float = 0.7,
) -> pd.DataFrame:

    df = df.copy()
    if bank_col not in df.columns:
        raise ValueError(f"Не знайдено колонку з назвою банку: '{bank_col}'")

    ordinal_levels = {}
    weights = {}

    for col, cfg in indicators_config.items():
        if col not in df.columns:
            raise ValueError(f"Вхідні дані не містять потрібної колонки: '{col}'")

        direction = cfg["direction"]
        higher_is_riskier = (direction == "riskier_when_higher")
        weights[col] = cfg.get("weight", 1.0)

        if cfg["method"] == "thresholds":
            t = cfg["thresholds"]
            cats = df[col].apply(lambda x: ordinalize_by_thresholds(x, t, higher_is_riskier=higher_is_riskier))
            ordinal_levels[col + "_ord"] = cats
        elif cfg["method"] == "quantiles":
            q = cfg["quantiles"]
            cats = ordinalize_by_quantiles(df[col], q, higher_is_riskier=higher_is_riskier)
            ordinal_levels[col + "_ord"] = cats
        else:
            raise ValueError(f"Невідомий method для '{col}': {cfg['method']}")

    ord_df = pd.DataFrame(ordinal_levels, index=df.index)

    w = np.array([weights[k] for k in indicators_config.keys()])
    w = w / w.sum()
    ord_cols = list(ord_df.columns)
    ord_mat = ord_df[ord_cols].values.astype(float)

    weighted_vals = []
    for i in range(ord_mat.shape[0]):
        row = ord_mat[i, :]
        mask = ~np.isnan(row)
        if not np.any(mask):
            weighted_vals.append(np.nan)
        else:
            ww = w[mask]
            ww = ww / ww.sum()
            weighted_vals.append(np.dot(row[mask], ww))

    ori_raw = pd.Series(weighted_vals, index=df.index, name="ORI_raw_1to5")

    ori_norm = (ori_raw - 1) / 4.0
    ori_norm.name = "ORI_0to1"

    hurwicz = compute_hurwicz(ord_df, alpha=hurwicz_alpha)

    risk_class = classify_tertiles(ori_norm, labels=("Low", "Medium", "High"))

    out = df[[bank_col]].copy()
    out = pd.concat([out, ord_df, ori_raw, ori_norm, hurwicz, risk_class], axis=1)

    out["rank_by_ori"] = out["ORI_0to1"].rank(method="min", ascending=False).astype(int)

    out = out.sort_values(by=["ORI_0to1", "hurwicz_risk"], ascending=False).reset_index(drop=True)
    return out


def demo_data() -> pd.DataFrame:
    data = {
        "bank": ["Bank A", "Bank B", "Bank C", "Bank D", "Bank E", "Bank F", "Bank G", "Bank H"],
        "car_pct": [12.5, 9.2, 15.1, 7.9, 13.3, 10.5, 11.0, 14.8],
        "npl_pct": [4.2, 9.8, 2.1, 12.5, 6.0, 7.5, 3.8, 1.9],
        "lcr_pct": [125, 90, 160, 75, 130, 105, 110, 150],
        "roa_pct": [1.2, 0.3, 1.8, -0.2, 0.9, 0.6, 0.8, 1.5],
        "ops_incidents_per_10k": [0.8, 2.5, 0.4, 3.2, 1.4, 1.8, 1.2, 0.6],
        "fail_rate_pct": [0.35, 0.9, 0.2, 1.2, 0.5, 0.6, 0.45, 0.25],
        "interbank_centrality": [0.45, 0.75, 0.30, 0.85, 0.60, 0.55, 0.40, 0.35],
    }
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Ordinal Risk Index for Banks/Payment Systems")
    parser.add_argument("--input", type=str, default=None, help="Вхідний CSV з показниками (якщо не задано — демо)")
    parser.add_argument("--output", type=str, default="ranked_banks.csv", help="Куди записати результати")
    parser.add_argument("--bank_col", type=str, default="bank", help="Назва колонки з іменами банків")
    parser.add_argument("--hurwicz_alpha", type=float, default=0.7, help="Альфа для критерію Гурвіца (0..1)")
    args = parser.parse_args()

    if args.input is None:
        print("  Вхідний файл не задано — використовую демо-дані.")
        df = demo_data()
    else:
        try:
            df = pd.read_csv(args.input)
        except Exception as e:
            print(f"Помилка читання CSV: {e}", file=sys.stderr)
            sys.exit(1)

    result = ordinal_risk_pipeline(
        df=df,
        indicators_config=INDICATORS_CONFIG,
        bank_col=args.bank_col,
        hurwicz_alpha=args.hurwicz_alpha,
    )

    # Вивід у консоль (топ-10)
    print("\nTOP ризиків (за ORI_0to1)")
    print(result[["bank", "ORI_0to1", "hurwicz_risk", "risk_class", "rank_by_ori"]].head(10).to_string(index=False))

    # Збереження повної таблиці
    try:
        result.to_csv(args.output, index=False)
        print(f"\nРезультати збережено у: {args.output}")
    except Exception as e:
        print(f"Помилка запису CSV: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
