"""Arabic Streamlit dashboard: real inference when model present; no fabricated detections."""
from __future__ import annotations
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from audit import POLICIES, read_scores, run_audit, stats, thresholds
from infer_audio import run as infer_real_audio, read_mono, aggregation_key

st.set_page_config(page_title="مِعيار الصوت", page_icon="🎙️", layout="wide")
st.markdown('''<style>
html, body, [data-testid="stApp"] {direction: rtl; text-align: right;}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {direction: ltr; text-align: center;}
[data-testid="stSidebar"] {direction: rtl;}
.block-container {max-width: 1180px; padding-top: 2rem;}
</style>''', unsafe_allow_html=True)
st.title("مِعيار الصوت | Audio Threshold Auditor")
st.caption("تحدي كشف التزييف الصوتي بالذكاء الاصطناعي · تدقيق العتبات قبل النشر · نموذج أولي V1")

one, two, three = st.tabs(["📊 تدقيق العتبات", "🎧 فحص ملف صوتي", "📋 منهجية واختبارات"])

with one:
    st.subheader("اختيار العتبة من المعايرة واختبار ثباتها")
    st.info("ارفع درجات فعلية من نموذج الكشف. **لا يُختار حد القرار اعتمادًا على بيانات الاختبار**. يوجد مثال مصطنع للتعلم داخل حزمة المشروع ولا يُعرض كأداء فعلي.")
    uploaded = st.file_uploader("ملف CSV (audio_id, split, label, bonafide_logit, aggregation, group_id)", type="csv", key="scores")
    if uploaded:
        try:
            df = read_scores(uploaded)
            mode = st.radio("سياسة التشغيل", POLICIES, horizontal=True, format_func=lambda x: {"conservative":"تقليل الإنذارات الخاطئة", "balanced":"توازن HTER", "sensitive":"حساسية مرتفعة"}[x])
            cf, cr = st.columns(2)
            with cf:
                max_fpr = st.slider("قيد FPR على المعايرة", 0.00, 0.50, 0.05, 0.01)
            with cr:
                min_recall = st.slider("قيد Recall على المعايرة", 0.50, 1.00, 0.95, 0.01)
            report = run_audit(df, mode, max_fpr, min_recall)
            st.success(f"العتبة الثابتة من مجموعة المعايرة: {report['threshold']:.6f} (درجة خام، ليست نسبة مئوية)")
            st.caption(f"النموذج: {report['model_id']} | تجميع الدرجات: {report['aggregation']}")
            c1, c2 = st.columns(2)
            for col, key, title in ((c1,'calibration','مجموعة المعايرة'),(c2,'test_fixed_threshold','اختبار مستقل بعتبة ثابتة')):
                m = report[key]
                with col:
                    st.markdown(f"#### {title}")
                    a,b,c=st.columns(3)
                    a.metric("F1",f"{m['f1']:.1%}")
                    b.metric("Recall",f"{m['recall']:.1%}")
                    c.metric("FPR",f"{m['fpr']:.1%}")
                    st.write(f"HTER: **{m['hter']:.2%}** | ROC-AUC: **{m['auc']:.3f}** | عينات: **{m['n']}**")
                    st.caption(f"95% CI للإنذارات الخاطئة: {m['fpr_ci95'][0]:.1%} – {m['fpr_ci95'][1]:.1%}")
            st.metric("فجوة نقل العتبة في HTER (اختبار − معايرة)",f"{report['hter_transfer_gap']:+.2%}")
            st.markdown('#### مقارنة مجالات الاختبار')
            if report.get('test_by_domain'):
                st.dataframe(pd.DataFrame.from_dict(report['test_by_domain'],orient='index'),use_container_width=True)
            else:
                st.caption('أضف عمود domain لقياس أثر اختلاف اللهجات أو ضغط التسجيلات.')
            st.markdown('#### منحنى الحساسية والإنذارات — للشرح فقط')
            # Plot on calibration only. The test split is never used for threshold exploration.
            cal = df[df.split == 'calibration']
            ts = np.unique(np.quantile(cal.spoof_score, np.linspace(0,1,61)))
            rows=[{'عتبة':t,'Recall':stats(cal.label,cal.spoof_score,t)['recall'],
                   'FPR':stats(cal.label,cal.spoof_score,t)['fpr']} for t in ts]
            st.line_chart(pd.DataFrame(rows).set_index('عتبة'))
            st.caption('هذا الرسم يستخدم بيانات المعايرة فقط. EER على الاختبار يظهر في التقرير كإحصاء وصفي، ولا يُستخدم لتحديد العتبة.')
            st.json(report, expanded=False)
            st.download_button("تنزيل تقرير التدقيق JSON", data=json.dumps(report, ensure_ascii=False, indent=2),file_name='miyar_threshold_report.json',mime='application/json')
            for note in report['notes']:
                st.warning(note)
        except Exception as exc:
            st.error(f"لم يمكن تدقيق الملف: {exc}")
    else:
        st.caption('لبدء التجربة: استخدم ملف examples/DEMO_SYNTHETIC_scores.csv أو ملف درجات فعلي. الملف التوضيحي ليس بيانات نموذج حقيقي.')

with two:
    st.subheader('فحص تسجيل عربي بنموذج DeepFense الرسمي')
    st.caption('هذه الصفحة لا تعرض نتائج مصطنعة. يتطلب الاستدلال الحقيقي تنزيل أوزان النموذج وتهيئة DeepFense على الجهاز الذي يستضيف التطبيق.')
    default_dir=os.getenv('MIYAR_MODEL_DIR','models/ArA-DF-Baseline')
    model_dir=st.text_input('مجلد النموذج على الخادم',value=default_dir)
    upload=st.file_uploader('اختر WAV أو FLAC (حد 120 ثانية في هذه النسخة)',type=['wav','flac'],key='audio')
    report_upload=st.file_uploader('اختياري: تقرير عتبة من تدقيق سابق (JSON)',type=['json'],key='threshold_report')
    required=[Path(model_dir)/name for name in ('best_model.pth','xlsr2_300m.pt','config.yaml','run_inference.py')]
    available=all(path.is_file() for path in required)
    if available:
        st.success('ملفات النموذج الأساسية موجودة. يحتاج التشغيل أيضًا إلى DeepFense واعتمادياته.')
    else:
        st.warning('أوزان النموذج أو ملف الاستدلال غير متاحة في المجلد المحدد. يمكن رفع ملف للاستماع، لكن لن يظهر حكم دون تشغيل النموذج.')
    if upload:
        data=upload.getvalue()
        if len(data)>25*1024*1024:
            st.error('الملف يتجاوز الحد التجريبي 25MB')
        else:
            st.audio(data,format='audio/wav' if upload.name.lower().endswith('.wav') else 'audio/flac')
            if st.button('تحليل الصوت فعليًا',type='primary',disabled=not available):
                with tempfile.TemporaryDirectory() as td:
                    path=Path(td)/('incoming.wav' if upload.name.lower().endswith('.wav') else 'incoming.flac')
                    path.write_bytes(data)
                    try:
                        with st.spinner('تشغيل نموذج الكشف على الجهاز المضيف...'):
                            reportpath=None
                            if report_upload:
                                reportpath=Path(td)/'threshold.json'
                                reportpath.write_bytes(report_upload.getvalue())
                            result=infer_real_audio(path,model_dir,decision_report=reportpath)
                        st.success('تمت معالجة الصوت بالنموذج')
                        st.write('**درجة الاشتباه الخام:**',f"{result['spoof_score']:.5f}")
                        st.write('**القرار:**',result['decision'])
                        st.caption('الدرجة ليست احتمال تزييف. النوافذ المشتبه بها إشارات استكشافية وليست دليلًا على حدود تغيير كلمة.')
                        st.dataframe(pd.DataFrame(result['segments']),use_container_width=True)
                        st.download_button('حفظ نتيجة الفحص JSON',json.dumps(result,ensure_ascii=False,indent=2),file_name='audio_result.json',mime='application/json')
                    except Exception as exc:
                        st.error(f'لم ينجح الاستدلال الحقيقي: {exc}')

with three:
    st.subheader('ماذا أُنجز، وما الذي لم يُثبت بعد؟')
    st.markdown('''
**أنجز في الكود:** معايرة عتبات بثلاث سياسات، منع تسرب group_id بين المعايرة والاختبار،
مؤشرات F1/Recall/FPR/HTER/AUC وفواصل ثقة، توليد درجات من ملفات صوتية موسومة (عند تهيئة النموذج)،
وفحص تسجيل فردي بالاعتماد على النموذج العربي الرسمي.

**لم يُثبت بعد:** نتيجة حقيقية على بيانات عربية مستقلة، نسبة كشف تشغيلية، دقة تحديد كلمة معدلة،
أو صلاحية العتبة عند اختلاف ظروف التسجيل.

**البروتوكول:** اجمع تسجيلات أصلية ومزيفة مولّدة بالذكاء الاصطناعي بإذن أصحابها،
افصل الأشخاص والمصادر ومشتقات الملفات بين المعايرة والاختبار، اختر العتبة أولًا، ثم اختبر دون تغييرها.
في النسخة الحالية `Sliding Window` ليس توطينًا جنائيًا مُختبَرًا.
''')
    st.code('python -m pytest -q\npython score_manifest.py --manifest my_manifest.csv --model-dir models/ArA-DF-Baseline --output scores_real.csv\npython audit.py --csv scores_real.csv --policy conservative --max-fpr 0.05',language='bash')
    st.markdown('**البيانات المطلوبة:**')
    st.code('audio_id,path,split,label,domain,group_id\na01,private_audio/a01.wav,calibration,0,clean,person01\na02,private_audio/a02.wav,test,1,compressed,person02',language='csv')
    st.caption('يجب أن تحتوي كل مجموعة على الصنفين. لا تنسخ نفس المتحدث أو تسجيله الأصلي والنسخة المعدلة إلى مجموعتين مختلفتين.')
