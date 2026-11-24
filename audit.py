"""Miyar Al Sawt: truthful threshold-transfer audits on fixed, disjoint splits.

Input labels are 0=bonafide, 1=AI-spoof; bonafide_logit is the official
ArA-DF baseline score (HIGHER = bonafide). We invert it for spoof_score.
Selecting thresholds on test labels is prohibited by design.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

REQUIRED = {"audio_id", "split", "label", "bonafide_logit", "aggregation"}
POLICIES = ("conservative", "balanced", "sensitive")


def read_scores(csv):
    df = pd.read_csv(csv, dtype={"audio_id": "str", "split": "str", "aggregation": "str", "group_id": "str"})
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"أعمدة CSV ناقصة: {', '.join(sorted(missing))}")
    if df.empty or df[list(REQUIRED)].isna().any().any():
        raise ValueError("الملف فارغ أو يحتوي خلايا أساسية فارغة")
    if df.audio_id.duplicated().any():
        raise ValueError("audio_id يجب أن يكون فريدًا عبر جميع المجموعات")
    if not df.split.isin(["calibration", "test"]).all() or set(df.split) != {"calibration", "test"}:
        raise ValueError("split يجب أن يتضمن calibration وtest فقط")
    labels = pd.to_numeric(df.label, errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("label يجب أن يكون 0=أصلي أو 1=مزيف")
    df["label"] = labels.astype(int)
    score = pd.to_numeric(df.bonafide_logit, errors="coerce")
    if score.isna().any() or not np.isfinite(score.to_numpy(dtype=float)).all():
        raise ValueError("الدرجات يجب أن تكون أرقامًا منتهية")
    df["bonafide_logit"] = score.astype(float)
    if df.aggregation.nunique() != 1 or not str(df.aggregation.iloc[0]).strip():
        raise ValueError("الملف يجب أن يستخدم طريقة تجميع aggregation واحدة معلنة")
    if "model_id" in df.columns and (df.model_id.isna().any() or df.model_id.nunique() != 1):
        raise ValueError("لا تمزج نتائج نماذج متعددة في تقرير عتبة واحد")
    if "group_id" in df.columns:
        if df.group_id.isna().any() or (df.group_id.str.strip() == "").any():
            raise ValueError("group_id يجب أن يكون غير فارغ عند استخدامه")
        split_counts = df.groupby("group_id").split.nunique()
        if (split_counts > 1).any():
            raise ValueError("تسرب بيانات: group_id مكرر بين المعايرة والاختبار")
    for split, part in df.groupby("split"):
        if set(part.label) != {0, 1}:
            raise ValueError(f"مجموعة {split} تحتاج أمثلة أصلية ومزيفة")
    df["spoof_score"] = -df.bonafide_logit
    return df


def stats(labels, scores, threshold):
    y, s = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if not len(y) or set(y.tolist()) != {0, 1} or not np.isfinite(s).all():
        raise ValueError("تقييم المجموعة يتطلب كلا الصنفين ودرجات صالحة")
    pred = s >= threshold
    tp = int(((y == 1) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": int(len(y)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": float(precision), "recall": float(recall), "fpr": float(fpr),
            "f1": float(f1), "hter": float((fpr + (1 - recall))/2)}


def thresholds(scores):
    unique = np.unique(np.asarray(scores, dtype=float))
    if not len(unique) or not np.isfinite(unique).all():
        raise ValueError("لا توجد درجات صالحة")
    return np.r_[np.nextafter(unique.max(), np.inf), unique]


def operating_points(labels, scores):
    """All empirical operating points in O(n log n), including alert-none."""
    y, s = np.asarray(labels,dtype=int), np.asarray(scores,dtype=float)
    if not len(y) or set(y.tolist()) != {0, 1} or not np.isfinite(s).all():
        raise ValueError("Both classes and finite scores required")
    order = np.argsort(-s, kind="stable")
    sorted_s, sorted_y = s[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_s) != 0), len(s)-1]
    ctp = np.cumsum(sorted_y == 1)[ends].astype(float)
    cfp = np.cumsum(sorted_y == 0)[ends].astype(float)
    positives, negatives = float((y==1).sum()), float((y==0).sum())
    return pd.DataFrame({"threshold": np.r_[np.nextafter(sorted_s[0], np.inf), sorted_s[ends]],
       "recall": np.r_[0.0, ctp/positives],
       "fpr": np.r_[0.0, cfp/negatives]})


def choose_threshold(labels, scores, policy="conservative", max_fpr=0.05, min_recall=0.95):
    """Select fixed threshold using CALIBRATION ONLY, never test."""
    if policy not in POLICIES or not (0 <= max_fpr <= 1 and 0 <= min_recall <= 1):
        raise ValueError("سياسة أو قيود عتبة غير صالحة")
    points = operating_points(labels,scores)
    fnr = 1.0 - points.recall.to_numpy()
    fpr = points.fpr.to_numpy()
    recall = points.recall.to_numpy()
    t = points.threshold.to_numpy()
    if policy == "conservative":
        candidates = np.where(fpr <= max_fpr + 1e-12)[0]
        ix = min(candidates, key=lambda i: (-recall[i], fpr[i], -t[i]))
    elif policy == "sensitive":
        candidates = np.where(recall >= min_recall - 1e-12)[0]
        ix = min(candidates, key=lambda i: (fpr[i], -recall[i], t[i]))
    else:
        ix = min(range(len(points)), key=lambda i: ((fpr[i]+fnr[i])/2, abs(fpr[i]-fnr[i]), -t[i]))
    return float(t[ix])


def retrospective_eer(labels, scores):
    """Approximate empirical EER: diagnostic on labeled split, NOT a deployable threshold."""
    pts = operating_points(labels,scores)
    delta = (pts.fpr - (1-pts.recall)).to_numpy()
    # Linear interpolation across adjacent operating points around FPR==FNR.
    ix = np.flatnonzero(delta >= 0)
    if len(ix) and ix[0] > 0:
        j = int(ix[0])
        prev, curr = j-1, j
        d0,d1 = delta[prev],delta[curr]
        lam = (-d0)/(d1-d0) if d1 != d0 else 0.0
        return float(pts.fpr.iloc[prev] + lam*(pts.fpr.iloc[curr]-pts.fpr.iloc[prev]))
    best = int(np.argmin(np.abs(delta)))
    return float((pts.fpr.iloc[best] + 1 - pts.recall.iloc[best])/2)


def binomial_interval(success, trials, alpha=0.05):
    """Exact Clopper-Pearson binomial confidence interval, descriptive only."""
    if trials < 1:
        return [None, None]
    lo = 0.0 if success == 0 else float(beta.ppf(alpha/2, success, trials-success+1))
    hi = 1.0 if success == trials else float(beta.ppf(1-alpha/2, success+1, trials-success))
    return [lo, hi]


def describe_split(df, threshold):
    m = stats(df.label, df.spoof_score, threshold)
    m["fpr_ci95"] = binomial_interval(m["fp"], m["fp"]+m["tn"])
    m["recall_ci95"] = binomial_interval(m["tp"], m["tp"]+m["fn"])
    m["auc"] = float(roc_auc_score(df.label, df.spoof_score))
    m["eer_retrospective_only"] = retrospective_eer(df.label, df.spoof_score)
    return m


def run_audit(df, policy="conservative", max_fpr=0.05, min_recall=0.95):
    cal = df[df.split == "calibration"]
    test = df[df.split == "test"]
    if len(cal) == 0 or len(test) == 0:
        raise ValueError("يلزم calibration وtest")
    threshold = choose_threshold(cal.label, cal.spoof_score, policy, max_fpr, min_recall)
    result = {"policy": policy, "threshold": threshold, "max_fpr_requested": max_fpr,
              "min_recall_requested": min_recall,
              "model_id": str(df.model_id.iloc[0]) if "model_id" in df else "unspecified",
              "aggregation": str(df.aggregation.iloc[0]),
              "score_definition": "spoof_score = -bonafide_logit; higher = more suspicious; NOT calibrated probability",
              "calibration": describe_split(cal, threshold),
              "test_fixed_threshold": describe_split(test, threshold)}
    result["hter_transfer_gap"] = result["test_fixed_threshold"]["hter"] - result["calibration"]["hter"]
    result["notes"] = ["العتبة مأخوذة من بيانات المعايرة فقط؛ الاختبار للتقييم لا لإعادة الضبط.",
                       "الـ EER قيمة وصفية محسوبة بعد الاطلاع على التسميات ولا تُستخدم لاختيار عتبة النشر."]
    if min((cal.label==0).sum(), (cal.label==1).sum()) < 30:
        result["notes"].append("عينة المعايرة صغيرة: لا تُفسّر بلوغ FPR منخفض كضمان تشغيلي.")
    if "domain" in df.columns:
        result["test_by_domain"] = {}
        for name, part in test.groupby("domain", dropna=False):
            if set(part.label) == {0, 1}:
                result["test_by_domain"][str(name)] = describe_split(part, threshold)
            else:
                result["test_by_domain"][str(name)] = {"n": int(len(part)), "note": "هذه الفئة لا تضم الصنفين؛ المقاييس غير متاحة"}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--policy", choices=POLICIES, default="conservative")
    p.add_argument("--max-fpr", type=float, default=0.05)
    p.add_argument("--min-recall", type=float, default=0.95)
    p.add_argument("--output", default="threshold_report.json")
    a = p.parse_args()
    report = run_audit(read_scores(a.csv), a.policy, a.max_fpr, a.min_recall)
    Path(a.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
