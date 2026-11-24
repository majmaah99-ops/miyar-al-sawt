"""CPU audio deepfake detector using a compact publicly available ONNX model.

Model: nakas/gpttrace-wavlm-audio-v3 (CC BY-SA 4.0).
As specified in the publisher's model card:
- 16 kHz mono float32 audio
- 4-second windows (64,000 samples)
- zero-mean, unit-standard-deviation normalization per window
- average logits (synthetic minus human), up to four evenly spaced windows

We never claim that the score is a calibrated probability or forensic proof.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MODEL_ID = "nakas/gpttrace-wavlm-audio-v3"
MODEL_FILE = "onnx/model_quantized.onnx"
REVISION = os.environ.get("MIYAR_LIGHT_MODEL_REVISION", "main")
SAMPLE_RATE = 16000
WINDOW = 64000
AGGREGATION = "gpttrace_mean_4_windows_v1"
MAX_DURATION = 120.0


def prepare_audio(file_path: str | Path) -> np.ndarray:
    audio, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
    if sr < 8000 or sr > 192000:
        raise ValueError("معدل العينات غير مدعوم")
    if len(audio) == 0 or len(audio) / sr > MAX_DURATION:
        raise ValueError("يجب أن يكون التسجيل غير فارغ ولا يزيد عن 120 ثانية")
    if not np.isfinite(audio).all():
        raise ValueError("الصوت يحتوي قيمًا غير صالحة")
    mono = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        from math import gcd
        factor = gcd(sr, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // factor, sr // factor)
    if not np.isfinite(mono).all() or float(np.std(mono)) < 1e-6:
        raise ValueError("التسجيل صامت أو غير مناسب للتحليل")
    return np.asarray(mono, dtype=np.float32)


def make_windows(audio: np.ndarray, count: int = 4):
    """Return time ranges and normalized [1, 64000] model inputs."""
    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim != 1 or not len(signal) or count < 1:
        raise ValueError("مدخل صوتي غير صالح")
    if len(signal) < WINDOW:
        signal = np.tile(signal, (WINDOW + len(signal) - 1) // len(signal))[:WINDOW]
        starts = [0]
    else:
        last = len(signal) - WINDOW
        starts = sorted(set(int(round(x)) for x in np.linspace(0, last, min(count, 4))))
    out = []
    for start in starts:
        segment = signal[start : start + WINDOW].copy()
        mean = float(segment.mean())
        std = float(segment.std())
        if std < 1e-6:
            continue
        segment = ((segment - mean) / std).astype(np.float32)
        out.append((round(start / SAMPLE_RATE, 3), round(min(start + WINDOW, len(audio)) / SAMPLE_RATE, 3), segment[None, :]))
    if not out:
        raise ValueError("المقاطع الصوتية لا تحتوي إشارة صالحة")
    return out


def _ai_minus_human(output) -> float:
    """Published two-class output: [human, AI]. Reject unknown ONNX shapes."""
    a = np.asarray(output, dtype=np.float64)
    if a.size != 2:
        raise ValueError(f"شكل خرج ONNX غير متوقع: {a.shape}")
    flat = a.reshape(-1)
    if not np.isfinite(flat).all():
        raise ValueError("النموذج أعاد درجات غير صالحة")
    return float(flat[1] - flat[0])


def load_session():
    """Load once per process; download ONNX weights only when actually needed."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=MODEL_ID, filename=MODEL_FILE, revision=REVISION)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def evaluate(audio_path, session):
    audio = prepare_audio(audio_path)
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"عدد مدخلات النموذج غير مدعوم: {len(inputs)}")
    key = inputs[0].name
    pieces = []
    for start, end, values in make_windows(audio):
        raw_output = session.run(None, {key: values})
        if not raw_output:
            raise RuntimeError("لا توجد مخرجات من النموذج")
        logit = _ai_minus_human(raw_output[0])
        pieces.append({"start_seconds": start, "end_seconds": end, "spoof_score": round(logit, 6)})
    score = float(np.mean([p["spoof_score"] for p in pieces]))
    return {
        "model": MODEL_ID, "model_id": MODEL_ID,
        "aggregation": AGGREGATION,
        "spoof_score": round(score, 6),
        "duration_seconds": round(len(audio) / SAMPLE_RATE, 3),
        "windows": pieces,
        "score_interpretation": "Raw synthetic-minus-human logit, not calibrated probability",
    }


def with_decision(result: dict, threshold_report=None):
    """No fabricated calibration: a zero-logit baseline is distinctly labelled provisional."""
    data = dict(result)
    if threshold_report is None:
        threshold = 0.0
        data["decision_type"] = "provisional_non_calibrated_zero_logit"
        data["warning"] = "هذه مقارنة أولية بعتبة صفر غير معايرة على العربية؛ ليست إثباتًا على تزييف الصوت"
    else:
        report = (json.loads(Path(threshold_report).read_text(encoding="utf-8"))
                  if isinstance(threshold_report, (str, Path)) else threshold_report)
        if report.get("model_id") != MODEL_ID or report.get("aggregation") != AGGREGATION:
            raise ValueError("تقرير JSON لا يطابق النموذج الحالي وطريقة تجميع الدرجات. لا تستخدم تقرير المثال الاصطناعي.")
        threshold = float(report["threshold"])
        if not math.isfinite(threshold):
            raise ValueError("عتبة JSON غير صالحة")
        data["decision_type"] = "fixed_threshold_from_calibration_report"
        data["warning"] = "القرار مؤشر إحصائي يحتاج تحققًا مستقلًا، وليس دليلًا جنائيًا"
    data["threshold"] = threshold
    data["decision"] = "مؤشرات اشتباه بالتزييف" if data["spoof_score"] >= threshold else "لم تظهر مؤشرات كافية للتزييف"
    return data
