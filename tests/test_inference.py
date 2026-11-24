import csv
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from infer_audio import (read_mono, windows, run, aggregation_key, TARGET_SR,WINDOW_SIZE)


def test_resampling_and_mono(tmp_path):
    sr=8000
    t=np.arange(sr)/sr
    audio=np.column_stack([np.sin(2*np.pi*440*t),np.sin(2*np.pi*440*t)])
    path=tmp_path/'test.wav'
    sf.write(path,audio,sr)
    mono=read_mono(path)
    assert mono.ndim==1
    assert len(mono)==TARGET_SR


def test_silence_rejected(tmp_path):
    path=tmp_path/'silence.wav'
    sf.write(path,np.zeros(16000),16000)
    with pytest.raises(ValueError,match='Silent'):
        read_mono(path)


def test_windowing_last_window_covered():
    audio=np.ones(WINDOW_SIZE*3+234)
    positions=list(windows(audio))
    assert positions[0][0]==0
    assert len(positions[-1][1])==WINDOW_SIZE
    assert abs(positions[-1][0] - (len(audio)-WINDOW_SIZE)/TARGET_SR)<1e-6


def test_aggregation_stride_label():
    assert aggregation_key(2.0)=='window_max_v1_w64600_stride32000'
    assert aggregation_key(1.0) != aggregation_key(2.0)


def test_model_missing_blocks_fake_prediction(tmp_path):
    path=tmp_path/'voice.wav'
    sf.write(path,np.ones(32000)*0.1,16000)
    with pytest.raises(FileNotFoundError):
        run(path,tmp_path/'not_a_model')


def test_official_command_adapter_without_real_model(tmp_path):
    """A fake external script tests integration only, NOT neural model quality."""
    mdir=tmp_path/'mock_model'
    mdir.mkdir()
    for name in ('best_model.pth','xlsr2_300m.pt','config.yaml'):
        (mdir/name).write_text('fixture only')
    script='''import argparse,csv
from pathlib import Path
p=argparse.ArgumentParser()
for arg in ('audio_dir','config','checkpoint','ext','batch_size','output'):
 p.add_argument('--'+arg)
a=p.parse_args()
with open(a.output,'w',newline='') as f:
 w=csv.writer(f);w.writerow(['audio_id','logit'])
 for file in sorted(Path(a.audio_dir).glob('*.wav')):
  w.writerow([file.stem,-0.5])
'''
    (mdir/'run_inference.py').write_text(script)
    path=tmp_path/'voice.wav'
    sf.write(path,np.sin(2*np.pi*330*np.arange(16000)/16000),16000)
    result=run(path,mdir)
    assert abs(result['spoof_score']-0.5)<.00001
    assert result['decision'].startswith('unavailable')
    assert result['aggregation']=='window_max_v1_w64600_stride32000'
    report={"threshold":.3,"aggregation":result['aggregation'],"model_id":result['model']}
    rp=tmp_path/'report.json'
    rp.write_text(json.dumps(report))
    assert run(path,mdir,decision_report=rp)['decision']=='suspected'
    report['model_id']='wrong_model'
    rp.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='model_id'):
        run(path,mdir,decision_report=rp)
