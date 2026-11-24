"""Create ACTUAL labeled per-file scores from a manifest and the official ArA-DF model.

Manifest CSV: audio_id,path,split,label,domain,group_id. For this starter, model
loads separately per recording: best for small pilots, not large dataset sweeps.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from infer_audio import run, aggregation_key


def build_scores(manifest_csv, model_dir, stride=2.0):
    df = pd.read_csv(manifest_csv, dtype={"audio_id":str,"group_id":str})
    must = {"audio_id", "path", "split", "label", "group_id"}
    if must - set(df.columns):
        raise ValueError(f"Manifest missing {sorted(must - set(df.columns))}")
    if df.empty or df.audio_id.duplicated().any() or df[sorted(must)].isna().any().any():
        raise ValueError("Manifest contains duplicates, missing values, or is empty")
    if not df.label.isin([0,1]).all() or not df.split.isin(["calibration","test"]).all():
        raise ValueError("Labels must be 0/1, split must be calibration/test")
    root = Path(manifest_csv).resolve().parent
    rows=[]
    for item in df.itertuples(index=False):
        audio_file=Path(item.path)
        if not audio_file.is_absolute():
            audio_file=root/audio_file
        if audio_file.suffix.lower() not in ('.wav','.flac') or not audio_file.is_file():
            raise FileNotFoundError(f"Not a WAV/FLAC file: {audio_file}")
        result=run(audio_file, model_dir, stride_seconds=stride)
        if not np.isfinite(result['spoof_score']):
            raise ValueError(f"Non-finite model score: {audio_file}")
        row={"audio_id":item.audio_id,"split":item.split,"label":int(item.label),
             "bonafide_logit":-result["spoof_score"],"aggregation":result["aggregation"],
             "model_id":result["model"],"group_id":item.group_id}
        if "domain" in df.columns:
            row["domain"]=getattr(item, "domain")
        rows.append(row)
    # Last defense: validate split integrity even before writing.
    from audit import read_scores
    from io import StringIO
    output=pd.DataFrame(rows)
    read_scores(StringIO(output.to_csv(index=False)))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True)
    p.add_argument('--model-dir',required=True)
    p.add_argument('--stride',type=float,default=2.0)
    p.add_argument('--output',default='scores_real.csv')
    a=p.parse_args()
    if a.stride <= 0:
        p.error('stride must be positive')
    df=build_scores(a.manifest,a.model_dir,a.stride)
    df.to_csv(a.output,index=False)
    print(f"Created {len(df)} REAL model scores in {a.output}")


if __name__=='__main__':
    main()
