"""Batch calibration export tests use a fake prediction session, not detector weights."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import score_light_manifest
from audit import read_scores, run_audit
from light_detector import MODEL_ID, AGGREGATION


def test_batch_export_compatible_with_audit(tmp_path, monkeypatch):
    records = []
    for i in range(8):
        path = tmp_path / f"file_{i}.wav"
        sf.write(str(path), np.sin(2*np.pi*220*np.arange(16000)/16000), 16000)
        records.append({"audio_id":f"rec{i}", "path":path.name,
                        "split":"calibration" if i<4 else "test", "label":i%2,
                        "group_id":f"spk{i}", "domain":"clean" if i<4 else "compressed"})
    manifest=tmp_path/"manifest.csv"
    pd.DataFrame(records).to_csv(manifest,index=False)
    def mock_evaluate(path,session):
        ix=int(Path(path).stem.split("_")[-1])
        return {"spoof_score": float(1 if ix%2 else -1)}
    monkeypatch.setattr(score_light_manifest,"evaluate",mock_evaluate)
    scores=score_light_manifest.score_manifest(manifest,object())
    from io import StringIO
    audit=run_audit(read_scores(StringIO(scores.to_csv(index=False))))
    assert audit["model_id"]==MODEL_ID
    assert audit["aggregation"]==AGGREGATION
    assert audit["test_fixed_threshold"]["n"]==4


def test_batch_export_reject_missing_manifest_columns(tmp_path):
    path=tmp_path/"bad.csv"
    pd.DataFrame({"audio_id":["1"]}).to_csv(path,index=False)
    with pytest.raises(ValueError,match="Missing"):
        score_light_manifest.score_manifest(path,object())
