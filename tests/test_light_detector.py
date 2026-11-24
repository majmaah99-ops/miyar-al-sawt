"""Compact detector preprocessing and integration tests (no model weights required)."""
import numpy as np
import pytest
import soundfile as sf

from light_detector import (
    SAMPLE_RATE, WINDOW, MODEL_ID, AGGREGATION, _ai_minus_human,
    make_windows, prepare_audio, evaluate, with_decision
)


def test_window_count_and_normalization():
    x = (np.sin(2 * np.pi * 200 * np.arange(12 * SAMPLE_RATE) / SAMPLE_RATE) * 0.2).astype("float32")
    ws = make_windows(x)
    assert len(ws) == 4
    assert ws[0][0] == 0
    assert ws[-1][1] == 12
    for start, end, arr in ws:
        assert arr.shape == (1, WINDOW)
        assert abs(float(arr.mean())) < 0.001
        assert abs(float(arr.std()) - 1) < 0.001


def test_short_audio_repeat_pad():
    x = np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE).astype("float32")
    ws = make_windows(x)
    assert len(ws) == 1
    assert ws[0][2].shape == (1, WINDOW)


def test_two_logits_orientation_and_invalid_shape():
    assert _ai_minus_human(np.array([[0.2, 1.2]])) == pytest.approx(1.0)
    assert _ai_minus_human(np.array([[0.6, -0.4]])) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="شكل"):
        _ai_minus_human(np.ones((1, 3)))


def test_wav_flac_resample(tmp_path):
    sr = 8000
    wav = np.sin(2*np.pi*220*np.arange(sr*2)/sr).astype("float32")
    file = tmp_path / "test.wav"
    sf.write(file, wav, sr)
    audio = prepare_audio(file)
    assert audio.dtype == np.float32
    assert len(audio) == SAMPLE_RATE*2


def test_silent_rejected(tmp_path):
    file = tmp_path / "silent.wav"
    sf.write(file, np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    with pytest.raises(ValueError, match="صامت"):
        prepare_audio(file)


def test_mock_session_real_adapter_path(tmp_path):
    class I:
        name = "input_values"
    class Session:
        def get_inputs(self):
            return [I()]
        def run(self, ignored, inputs):
            assert inputs["input_values"].shape == (1, WINDOW)
            return [np.array([[0.0, 2.0]],dtype=np.float32)]
    wav = np.sin(2*np.pi*330*np.arange(SAMPLE_RATE*5)/SAMPLE_RATE).astype("float32")
    path = tmp_path / "voice.wav"
    sf.write(path, wav, SAMPLE_RATE)
    result = evaluate(path, Session())
    assert result["model"] == MODEL_ID
    assert result["aggregation"] == AGGREGATION
    assert result["spoof_score"] == pytest.approx(2)
    assert len(result["windows"]) == 4
    provisional = with_decision(result)
    assert provisional["decision_type"] == "provisional_non_calibrated_zero_logit"
    assert provisional["decision"] == "مؤشرات اشتباه بالتزييف"
    calibrated = with_decision(result, {"model_id": MODEL_ID, "aggregation": AGGREGATION, "threshold": 3})
    assert calibrated["decision_type"] == "fixed_threshold_from_calibration_report"
    assert calibrated["decision"] == "لم تظهر مؤشرات كافية للتزييف"


def test_reject_synthetic_or_mismatched_calibration():
    result = {"spoof_score": 2.0}
    with pytest.raises(ValueError, match="لا يطابق"):
        with_decision(result, {"model_id":"DEMO_SYNTHETIC_NO_REAL_MODEL","aggregation":"whole_utterance","threshold":1})
    with pytest.raises(ValueError, match="غير صالحة"):
        with_decision(result, {"model_id": MODEL_ID, "aggregation": AGGREGATION, "threshold": float("nan")})
