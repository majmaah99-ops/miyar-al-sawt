import io
import json
import numpy as np
import pandas as pd
import pytest
from audit import (read_scores, stats, choose_threshold, run_audit,
                   binomial_interval, operating_points, retrospective_eer)


def make_data():
    n=12
    return pd.DataFrame({
        'audio_id':[f'id_{i}' for i in range(n)],
        'split':['calibration']*6 + ['test']*6,
        'label':[0,0,0,1,1,1]*2,
        'bonafide_logit':[-.1,-.2,-.4,-.5,-.75,-.9,-.1,-.3,-.6,-.4,-.8,-.95],
        'aggregation':['whole_utterance']*n,
        'model_id':['model_a']*n,
        'domain':['clean']*6 + ['compressed']*6,
        'group_id':[f'group_{i}' for i in range(n)]})


def read(df):
    return read_scores(io.StringIO(df.to_csv(index=False)))


def test_stats_orientation():
    m=stats([0,0,1,1],[.1,.2,.8,.9],.5)
    assert (m['tp'],m['tn'],m['fp'],m['fn']) == (2,2,0,0)
    assert (m['f1'],m['hter']) == (1.0,0.0)


def test_calibration_policy_respects_fpr():
    y=[0,0,0,1,1]
    s=[.1,.3,.8,.5,.9]
    t=choose_threshold(y,s,'conservative',max_fpr=0)
    assert stats(y,s,t)['fpr']==0
    assert stats(y,s,t)['recall']==.5


def test_sensitive_policy_recall():
    y=[0,0,1,1]
    s=[.05,.45,.4,.8]
    t=choose_threshold(y,s,'sensitive',min_recall=1)
    assert stats(y,s,t)['recall']==1


def test_no_test_leakage_even_if_test_scores_changed():
    data=read(make_data())
    t1=run_audit(data)['threshold']
    data.loc[data.split=='test','spoof_score'] += 100000
    assert run_audit(data)['threshold']==t1


def test_leakage_repeated_group_fails():
    data=make_data()
    data.loc[6,'group_id']=data.loc[0,'group_id']
    with pytest.raises(ValueError,match='تسرب'):
        read(data)


def test_mixed_model_fails():
    data=make_data()
    data.loc[7,'model_id']='different'
    with pytest.raises(ValueError,match='نماذج'):
        read(data)


def test_mixed_aggregation_fails():
    data=make_data()
    data.loc[7,'aggregation']='max_window'
    with pytest.raises(ValueError,match='تجميع'):
        read(data)


def test_invalid_scores_fail():
    data=make_data()
    data.loc[0,'bonafide_logit']=np.inf
    with pytest.raises(ValueError):
        read(data)


def test_duplicate_audio_fails():
    data=make_data()
    data.loc[1,'audio_id']=data.loc[0,'audio_id']
    with pytest.raises(ValueError):
        read(data)


def test_confidence_interval_warns_implicit_small_sample():
    a,b=binomial_interval(0,10)
    assert a==0
    assert b>0.2


def test_auc_eer_in_report():
    report=run_audit(read(make_data()))
    assert 0<=report['calibration']['auc']<=1
    assert 0<=report['test_fixed_threshold']['eer_retrospective_only']<=1
    assert report['aggregation']=='whole_utterance'
    assert 'hter_transfer_gap' in report


def test_optimized_threshold_calculation():
    y=np.array([0,1,0,1,0,1,1,0])
    scores=np.array([.1,.2,.2,.4,.7,.8,.8,.9])
    pts=operating_points(y,scores)
    for row in pts.itertuples():
        result=stats(y,scores,row.threshold)
        assert abs(result['recall']-row.recall)<1e-10
        assert abs(result['fpr']-row.fpr)<1e-10
    assert 0<=retrospective_eer(y,scores)<=1


def test_report_json_finite():
    report=run_audit(read(make_data()))
    json.dumps(report,allow_nan=False)


def test_bad_schema_fails():
    with pytest.raises(ValueError):
        read_scores(io.StringIO('audio_id,label\na,0'))
