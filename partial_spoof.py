"""Experimental partial speech tampering inspection, not forensic proof.

Two methods:
  1) Existing neural classifier on overlapping four-second windows (no Arabic
     or word-level localization validation).
  2) Align short-time spectral fingerprints against a TRUSTED original file
     and flag acoustic differences, not their cause.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from light_detector import SAMPLE_RATE, WINDOW, MODEL_ID, _ai_minus_human, prepare_audio

SCAN_STEP_SECONDS = 2.0
REFERENCE_MAX_SECONDS = 45
REFERENCE_HOP_SECONDS = 0.1


def scan_windows(audio_path, session, step_seconds=SCAN_STEP_SECONDS, max_windows=30):
    """Return window evidence; no assertion that a word was cloned or edited."""
    if not 0.25 <= step_seconds <= 4:
        raise ValueError("خطوة المسح يجب أن تكون بين 0.25 و4 ثوانٍ")
    audio = prepare_audio(audio_path)
    if len(audio) <= WINDOW:
        starts = [0]
    else:
        hop = int(round(step_seconds * SAMPLE_RATE))
        last = len(audio) - WINDOW
        starts = list(range(0, last + 1, hop))
        if starts[-1] != last:
            starts.append(last)
    if len(starts) > max_windows:
        raise ValueError("عدد نوافذ المسح كبير جدًا. اختر تسجيلًا أقصر أو زد خطوة المسح")
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError("تنسيق مدخلات نموذج الكشف غير متوقع")
    key = inputs[0].name
    windows = []
    for start in starts:
        segment = audio[start:start + WINDOW]
        if len(segment) < WINDOW:
            segment = np.tile(segment, (WINDOW + len(segment) - 1) // len(segment))[:WINDOW]
        std = float(np.std(segment))
        if std < 1e-6:
            continue
        normalized = ((segment - float(np.mean(segment))) / std).astype("float32")[None, :]
        outputs = session.run(None, {key: normalized})
        if not outputs:
            raise ValueError("لم يرجع النموذج أي درجات")
        score = _ai_minus_human(outputs[0])
        windows.append({"start_seconds": round(start / SAMPLE_RATE, 2),
                        "end_seconds": round(min((start + WINDOW) / SAMPLE_RATE,
                                                 len(audio) / SAMPLE_RATE), 2),
                        "spoof_score": round(float(score), 5)})
    if not windows:
        raise ValueError("المقطع صامت أو لا يحتوي مقاطع صوتية صالحة")
    return {"model_id": MODEL_ID, "duration_seconds": round(len(audio) / SAMPLE_RATE, 2),
            "window_seconds": WINDOW / SAMPLE_RATE, "scan_step_seconds": step_seconds,
            "windows": windows,
            "score_definition": "AI minus human logit, NOT a probability",
            "warning": "درجات النوافذ استكشافية. هذا الكاشف غير مدرب على تحديد الكلمات المبدلة، وليس دليلًا على التزييف الجزئي."}


def _fingerprints(audio):
    """Gain-normalized short-time log-spectrum with ~200ms temporal resolution."""
    hop = round(REFERENCE_HOP_SECONDS * SAMPLE_RATE)
    frame = round(.20 * SAMPLE_RATE)
    x = np.asarray(audio, dtype=np.float32)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    x = np.ascontiguousarray(x)
    segments = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    taper = np.hanning(frame).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(segments * taper, axis=1))
    freqs = np.fft.rfftfreq(frame, d=1 / SAMPLE_RATE)
    edges = np.geomspace(110, 6000, 29)
    features = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        features.append(np.log1p(np.mean(spectrum[:, mask], axis=1)))
    f = np.column_stack(features).astype("float32")
    f -= f.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(f, axis=1, keepdims=True)
    # Silent frames have a zero feature vector; cosine distance is NaN there.
    # Use a fixed unit reference to keep alignment finite, without claiming
    # that silence is evidence of AI editing.
    f[norm[:, 0] < 1e-6, 0] = 1.0
    norm = np.linalg.norm(f, axis=1, keepdims=True)
    return f / np.maximum(norm, 1e-6)


def _dtw_match(a, b):
    """Dynamic time warping for short related recordings, with insertion penalty."""
    cost = cdist(a, b, metric="cosine").astype("float32")
    n, m = cost.shape
    total = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    moves = np.zeros((n + 1, m + 1), dtype=np.uint8)
    total[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = total[i - 1, j - 1]
            vert = total[i - 1, j] + .035
            horiz = total[i, j - 1] + .035
            if diag <= vert and diag <= horiz:
                total[i, j] = cost[i - 1, j - 1] + diag
            elif vert <= horiz:
                total[i, j] = cost[i - 1, j - 1] + vert
                moves[i, j] = 1
            else:
                total[i, j] = cost[i - 1, j - 1] + horiz
                moves[i, j] = 2
    i, j = n, m
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = moves[i, j]
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    if i or j:
        raise ValueError("لم ينجح محاذاة التسجيلين")
    scores = np.zeros(m, dtype=np.float64)
    counts = np.zeros(m, dtype=np.int32)
    for i, j in path:
        scores[j] += float(cost[i, j])
        counts[j] += 1
    return scores / np.maximum(counts, 1)


def _events(times, scores, cutoff, minimum_frames=2):
    flags = scores > cutoff
    for i in range(1, len(flags) - 1):
        if flags[i - 1] and flags[i + 1]:
            flags[i] = True
    output = []
    i = 0
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(flags) and flags[j + 1]:
            j += 1
        if j - i + 1 >= minimum_frames:
            output.append({"start_seconds": round(float(times[i]), 2),
                           "end_seconds": round(float(times[j] + REFERENCE_HOP_SECONDS + .10), 2),
                           "peak_difference": round(float(np.max(scores[i:j + 1])), 3)})
        i = j + 1
    return output


def compare_to_reference(original_path, suspicious_path):
    """Only identify discrepancies between versions of same recording.

    Any mismatch (compression/noise/trim/splice) may be flagged; NOT proof of AI.
    """
    original = prepare_audio(original_path)
    suspect = prepare_audio(suspicious_path)
    if max(len(original), len(suspect)) / SAMPLE_RATE > REFERENCE_MAX_SECONDS:
        raise ValueError("مقارنة المرجع تدعم تسجيلات حتى 45 ثانية في هذه النسخة")
    discrepancies = _dtw_match(_fingerprints(original), _fingerprints(suspect))
    baseline = float(np.median(discrepancies))
    mad = float(np.median(np.abs(discrepancies - baseline)))
    cutoff = max(.16, baseline + 5 * 1.4826 * mad)
    times = np.arange(len(discrepancies)) * REFERENCE_HOP_SECONDS
    return {"original_duration_seconds": round(len(original) / SAMPLE_RATE, 2),
            "suspicious_duration_seconds": round(len(suspect) / SAMPLE_RATE, 2),
            "method": "DTW of normalized short-time log-spectrum fingerprints",
            "hop_seconds": REFERENCE_HOP_SECONDS, "frame_seconds": .20,
            "baseline_median": round(baseline, 4), "heuristic_cutoff": round(cutoff, 4),
            "regions": _events(times, discrepancies, cutoff),
            "timeline": [{"time_seconds": round(float(t), 2),
                          "difference_index": round(float(s), 4)}
                         for t, s in zip(times, discrepancies)],
            "warning": "هذه مقارنة اختلافات صوتية بين نسخة مشتبه بها ومرجع موثوق، وليست إثباتًا أن الجزء مولّد بالذكاء الاصطناعي. الاختلاف قد ينشأ عن القص أو إعادة الترميز أو الضوضاء، والدقة الزمنية تقريبية."}
