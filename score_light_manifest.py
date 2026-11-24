"""Batch-score labeled recordings with the compact ONNX model.

Manifest columns: audio_id,path,split,label,group_id[,domain].
Split subjects/recording families across calibration/test before running this.
Results use raw logits only and are compatible with audit.py.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from audit import read_scores
from light_detector import load_session, evaluate, MODEL_ID, AGGREGATION


def score_manifest(path, session):
    manifest = Path(path).resolve()
    df = pd.read_csv(manifest, dtype={"audio_id":str,"group_id":str,"split":str})
    required = {"audio_id","path","split","label","group_id"}
    if required - set(df):
        raise ValueError("Missing columns: " + ", ".join(sorted(required-set(df))))
    if df.empty or df[list(required)].isna().any().any() or df.audio_id.duplicated().any():
        raise ValueError("Invalid manifest, missing values, or repeated audio_id")
    if not set(df.label.unique()).issubset({0,1}) or set(df.split.unique()) != {"calibration","test"}:
        raise ValueError("Need 0/1 labels and calibration/test splits")
    rows = []
    for rec in df.to_dict("records"):
        file = Path(rec["path"])
        if not file.is_absolute():
            file = manifest.parent / file
        if not file.is_file() or file.suffix.lower() not in (".wav", ".flac"):
            raise FileNotFoundError(str(file))
        result = evaluate(file, session)
        row = {"audio_id": rec["audio_id"], "split": rec["split"],
               "label": int(rec["label"]), "bonafide_logit": -result["spoof_score"],
               "aggregation": AGGREGATION, "model_id": MODEL_ID, "group_id": rec["group_id"]}
        if "domain" in rec:
            row["domain"] = rec["domain"]
        rows.append(row)
    scores = pd.DataFrame(rows)
    from io import StringIO
    read_scores(StringIO(scores.to_csv(index=False)))  # validate group leakage + both classes
    return scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True)
    p.add_argument("--output",default="scores_light_real.csv")
    a=p.parse_args()
    session=load_session()
    df=score_manifest(a.manifest,session)
    df.to_csv(a.output,index=False)
    print(f"Scored {len(df)} real files: {a.output}")


if __name__=="__main__":
    main()
