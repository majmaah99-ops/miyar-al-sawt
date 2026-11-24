import pandas as pd
import numpy as np
import soundfile as sf
import pytest

from audit import read_scores, run_audit
import score_manifest


def test_manifest_builds_real_schema_with_inference_adapter(monkeypatch,tmp_path):
    """Mock inference scores to exercise integration; does not test detector quality."""
    paths=[]
    for i in range(8):
        path=tmp_path/f'file_{i}.wav'
        sf.write(path,np.sin(2*np.pi*330*np.arange(8000)/16000),16000)
        paths.append(path)
    df=pd.DataFrame({'audio_id':[f'clip{i}' for i in range(8)],
        'path':[str(p) for p in paths],
        'split':['calibration']*4+['test']*4,
        'label':[0,0,1,1,0,0,1,1],
        'domain':['clean']*4+['compressed']*4,
        'group_id':[f'sp{i}' for i in range(8)]})
    manifest=tmp_path/'manifest.csv'
    df.to_csv(manifest,index=False)
    def fake_infer(path,model_dir,stride_seconds=2.0):
        i=int(path.stem.split('_')[1])
        score=[-2,-1,1,2,-1.7,-.8,.6,1.8][i]
        return {'spoof_score':score,'aggregation':'window_max_v1_w64600_stride32000',
                'model':'ArabicSpeech/ArA-DF-Baseline'}
    monkeypatch.setattr(score_manifest,'run',fake_infer)
    output=score_manifest.build_scores(manifest,'some-model')
    result=run_audit(read_scores(__import__('io').StringIO(output.to_csv(index=False))))
    assert result['test_fixed_threshold']['n']==4
    assert result['model_id']=='ArabicSpeech/ArA-DF-Baseline'


def test_manifest_rejects_invalid_labels(tmp_path):
    file=tmp_path/'manifest.csv'
    pd.DataFrame({'audio_id':['x'],'path':['x.wav'],'split':['test'],'label':[4],'group_id':['g']}).to_csv(file,index=False)
    with pytest.raises(ValueError,match='Labels'):
        score_manifest.build_scores(file,'model')
