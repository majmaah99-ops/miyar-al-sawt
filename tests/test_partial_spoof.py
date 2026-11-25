"""Mechanical integration tests. These synthetic tones do not prove Arabic deepfake detection."""
import numpy as np
import pytest
import soundfile as sf
from partial_spoof import scan_windows, compare_to_reference, SAMPLE_RATE

class MockInput:
    name = "waveform"

class MockSession:
    def get_inputs(self):
        return [MockInput()]
    def run(self, unused, inputs):
        assert inputs["waveform"].shape == (1, 64000)
        return [np.array([[0.0, 1.0]], dtype=np.float32)]

def tone(seconds=6):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.12 * np.sin(2 * np.pi * 210 * t)
            + 0.07 * np.sin(2 * np.pi * 400 * t)
            + 0.04 * np.sin(2 * np.pi * (700 + 200 * np.sin(t)) * t)).astype("float32")

def test_scan_overlapping_windows(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, tone(7), SAMPLE_RATE)
    result = scan_windows(p, MockSession(), step_seconds=1)
    assert len(result["windows"]) == 4
    assert result["windows"][0]["spoof_score"] == 1.0

def test_scan_invalid_step(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, tone(), SAMPLE_RATE)
    with pytest.raises(ValueError):
        scan_windows(p, MockSession(), step_seconds=.1)

def test_reference_identical_has_no_events(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, tone(5), SAMPLE_RATE)
    result = compare_to_reference(p, p)
    assert result["regions"] == []
    assert result["baseline_median"] < .001

def test_synthetic_replaced_interval_is_flagged(tmp_path):
    original = tone(6)
    modified = original.copy()
    i, j = int(3 * SAMPLE_RATE), int(4 * SAMPLE_RATE)
    t = np.arange(j - i) / SAMPLE_RATE
    modified[i:j] = (0.15 * np.sin(2 * np.pi * 1500 * t)
                     + 0.1 * np.sin(2 * np.pi * 2200 * t)).astype("float32")
    a, b = tmp_path / "original.wav", tmp_path / "edited.wav"
    sf.write(a, original, SAMPLE_RATE)
    sf.write(b, modified, SAMPLE_RATE)
    result = compare_to_reference(a, b)
    assert result["regions"], result
    assert any(x["start_seconds"] <= 3.5 <= x["end_seconds"]
               for x in result["regions"]), result["regions"]

def test_reference_max_duration(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, tone(46), SAMPLE_RATE)
    with pytest.raises(ValueError, match="45"):
        compare_to_reference(p, p)


def test_speech_with_silent_pauses_is_finite(tmp_path):
    x = tone(5)
    x[:int(.5 * SAMPLE_RATE)] = 0
    x[-int(.5 * SAMPLE_RATE):] = 0
    p = tmp_path / "with_pauses.wav"
    sf.write(p, x, SAMPLE_RATE)
    result = compare_to_reference(p, p)
    assert np.isfinite([item["difference_index"] for item in result["timeline"]]).all()
    assert result["regions"] == []
