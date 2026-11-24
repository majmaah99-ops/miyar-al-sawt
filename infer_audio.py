"""Run official ArA-DF-Baseline inference on one WAV/FLAC file.

Requires original DeepFense model files downloaded separately from Hugging Face.
Does not invent a score or give a calibrated 'probability'.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

TARGET_SR = 16000
WINDOW_SIZE = 64600  # matches official config (4.0375 s)
MAX_DURATION_S = 120  # pilot safety cap; avoids arbitrary-long uploads

def aggregation_key(stride_seconds):
    step = max(1, round(float(stride_seconds) * TARGET_SR))
    return f"window_max_v1_w{WINDOW_SIZE}_stride{step}"


def read_mono(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if len(audio) == 0:
        raise ValueError("Empty audio file")
    if sr < 8000 or sr > 192000:
        raise ValueError("Unsupported sample rate; expected 8000–192000 Hz")
    if len(audio) / sr > MAX_DURATION_S:
        raise ValueError(f"Audio exceeds {MAX_DURATION_S} seconds for this pilot")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples")
    if float(np.max(np.abs(audio))) < 1e-8:
        raise ValueError("Silent audio; no speech to evaluate")
    audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(int(sr), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, int(sr) // g)
    return audio.astype("float32")


def windows(audio, stride_seconds=2.0):
    stride = max(1, round(stride_seconds * TARGET_SR))
    if len(audio) <= WINDOW_SIZE:
        yield 0.0, audio
        return
    starts = list(range(0, len(audio) - WINDOW_SIZE + 1, stride))
    last = len(audio) - WINDOW_SIZE
    if starts[-1] != last:
        starts.append(last)
    for start in starts:
        yield start / TARGET_SR, audio[start:start+WINDOW_SIZE]


def run(audio_path, model_dir, stride_seconds=2.0, decision_report=None):
    if not 0 < stride_seconds <= WINDOW_SIZE / TARGET_SR:
        raise ValueError("Stride must be > 0 and <= window duration")
    mdir = Path(model_dir).expanduser().resolve()
    required = [mdir / name for name in ("best_model.pth", "xlsr2_300m.pt", "config.yaml", "run_inference.py")]
    if not all(p.is_file() for p in required):
        raise FileNotFoundError("Missing model files: " + ", ".join(str(p) for p in required if not p.is_file()))
    audio = read_mono(audio_path)
    positions = []
    with tempfile.TemporaryDirectory() as tempdir:
        td = Path(tempdir)
        for i, (start, chunk) in enumerate(windows(audio, stride_seconds)):
            fname = f"segment_{i:05d}.wav"
            sf.write(td / fname, chunk, TARGET_SR)
            positions.append({"audio_id": Path(fname).stem, "start_seconds": round(start, 3),
                              "end_seconds": round(min(start + WINDOW_SIZE / TARGET_SR, len(audio) / TARGET_SR), 3)})
        output = td / "scores.csv"
        cmd = [sys.executable, str(mdir / "run_inference.py"),
               "--audio_dir", str(td), "--config", str(mdir / "config.yaml"),
               "--checkpoint", str(mdir / "best_model.pth"),
               "--ext", "wav", "--batch_size", "8", "--output", str(output)]
        subprocess.run(cmd, check=True)
        with output.open(newline="", encoding="utf-8") as stream:
            scores = {row["audio_id"]: float(row["logit"]) for row in csv.DictReader(stream)}
    for seg in positions:
        if seg["audio_id"] not in scores:
            raise RuntimeError(f"No score for {seg['audio_id']}")
        seg["bonafide_logit"] = scores[seg["audio_id"]]
        if not math.isfinite(seg["bonafide_logit"]):
            raise ValueError("Non-finite model output")
        seg["spoof_score"] = -scores[seg["audio_id"]]  # orientation from official inference script
        del seg["audio_id"]
    # Maximum pooled score across windows is a heuristic; must calibrate this
    # same exact aggregation on the calibration split before deployment.
    overall = max(s["spoof_score"] for s in positions)
    result = {"source": str(audio_path), "model": "ArabicSpeech/ArA-DF-Baseline",
              "spoof_score": overall, "aggregation": aggregation_key(stride_seconds),
              "score_interpretation": "raw model score; NOT a calibrated probability",
              "duration_seconds": round(len(audio)/TARGET_SR, 3), "segments": positions,
              "decision": "unavailable — calibrate a file-level threshold first"}
    if decision_report:
        report = json.loads(Path(decision_report).read_text(encoding="utf-8"))
        if report.get("aggregation") != result["aggregation"]:
            raise ValueError("Threshold report aggregation does not match inference. Do not mix file and window thresholds.")
        if report.get("model_id") != result["model"]:
            raise ValueError("Threshold model_id differs from active model")
        result["decision"] = "suspected" if overall >= float(report["threshold"]) else "below calibrated alert threshold"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", required=True, help="WAV or FLAC")
    p.add_argument("--model-dir", required=True, help="Folder containing official files")
    p.add_argument("--stride", type=float, default=2.0)
    p.add_argument("--threshold-report", help="Optional fixed threshold report with matching aggregation")
    p.add_argument("--output", default="single_audio_result.json")
    a = p.parse_args()
    if a.stride <= 0:
        p.error("stride must be positive")
    report = run(a.audio, a.model_dir, a.stride, a.threshold_report)
    Path(a.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
