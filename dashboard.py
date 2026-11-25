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
from infer_audio import run as infer_real_audio
from light_detector import load_session as load_light_session, evaluate as evaluate_light_audio, with_decision as light_decision
from partial_spoof import scan_windows as scan_partial_windows, compare_to_reference

@st.cache_resource(show_spinner=False)
def cached_light_model():
    return load_light_session()


st.set_page_config(page_title="مِعيار الصوت", page_icon="🎙️", layout="wide")
st.markdown('''<style>
html, body, [data-testid="stApp"] {direction: rtl; text-align: right;}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {direction: ltr; text-align: center;}
[data-testid="stSidebar"] {direction: rtl;}
.block-container {max-width: 1180px; padding-top: 2rem;}
</style>''', unsafe_allow_html=True)
st.title("مِعيار الصوت | Audio Threshold Auditor")
st.caption("تحدي كشف التزييف الصوتي بالذكاء الاصطناعي · تدقيق العتبات قبل النشر · نموذج أولي V1")

one, two, three, four = st.tabs(["📊 تدقيق العتبات", "🎧 فحص ملف صوتي", "🔎 فحص التلاعب الجزئي", "📋 منهجية واختبارات"])

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
    st.subheader('فحص تسجيل صوتي بنموذج ذكاء اصطناعي حقيقي')
    st.info('أصبح الفحص المبدئي متاحًا عبر نموذج ONNX مضغوط يعمل على CPU. يتم تنزيل النموذج مرة واحدة عند أول تحليل؛ ولا يتطلب ملف JSON.')
    st.caption('النموذج GPTTrace WavLM (122MB، CC BY-SA 4.0) مدرّب على بيانات يغلب عليها الإنجليزية؛ نتائجه على العربية غير مثبتة بعد. لا تُستخدم النتائج لإثبات صحة التسجيل أو اتهام شخص.')
    backend = st.radio('طريقة التحليل', ['نموذج سحابي خفيف — تجربة فورية', 'النموذج العربي الكبير — يتطلب إعداد خادم'], horizontal=False)
    upload = st.file_uploader('ارفع تسجيل WAV أو FLAC (أقل من 25 ميجابايت و120 ثانية)', type=['wav', 'flac'], key='audio')
    report_upload = st.file_uploader('اختياري: تقرير معايرة عتبة مناسب لهذا النموذج (JSON)؛ اتركه فارغًا للتجربة الأولى', type=['json'], key='threshold_report')
    if backend.startswith('نموذج سحابي'):
        st.caption('تُستخدم عتبة صفر غير معايرة عند عدم تقديم تقرير متوافق. درجة النموذج خام وليست احتمالًا مئويًا.')
        if upload:
            audio_bytes = upload.getvalue()
            if len(audio_bytes) > 25 * 1024 * 1024:
                st.error('حجم الملف أكبر من 25 ميجابايت')
            else:
                st.audio(audio_bytes, format='audio/wav' if upload.name.lower().endswith('.wav') else 'audio/flac')
                if st.button('تحليل الصوت فعليًا', type='primary', key='run_real_light'):
                    with tempfile.TemporaryDirectory() as td:
                        suffix = '.flac' if upload.name.lower().endswith('.flac') else '.wav'
                        audio_path = Path(td) / ('uploaded' + suffix)
                        audio_path.write_bytes(audio_bytes)
                        try:
                            with st.spinner('تحميل نموذج الكشف عند الحاجة وتحليل الصوت على CPU...'):
                                session = cached_light_model()
                                result = evaluate_light_audio(audio_path, session)
                                calibrated_report = json.loads(report_upload.getvalue().decode('utf-8')) if report_upload else None
                                result = light_decision(result, calibrated_report)
                            st.success('تم تشغيل نموذج كشف حقيقي وإنتاج درجات استدلال')
                            st.markdown('### نتيجة الفحص الأولية')
                            st.write('**القرار:**', result['decision'])
                            st.metric('درجة الاشتباه الخام', f"{result['spoof_score']:.4f}")
                            st.caption(f"العتبة: {result['threshold']:.4f} | نوع القرار: {result['decision_type']}")
                            st.warning(result['warning'])
                            st.markdown('#### النوافذ الصوتية التي استخدمها النموذج')
                            st.dataframe(pd.DataFrame(result['windows']), use_container_width=True, hide_index=True)
                            st.caption('قيم النوافذ لا تعني تحديد كلمات مزيفة أو لحظة تركيب الصوت بدقة.')
                            st.download_button('تنزيل تقرير الفحص JSON', data=json.dumps(result, ensure_ascii=False, indent=2),
                                               file_name='miyar_audio_detection.json', mime='application/json')
                        except Exception as exc:
                            st.error(f'تعذر التحليل الحقيقي: {exc}')
                            st.info('إذا ظهر خطأ تنزيل الأوزان أو نفاد الذاكرة، تحقق من موارد Streamlit والسجل في Manage app. يمكنك تشغيل النموذج على خادم أقوى عند الحاجة.')
        else:
            st.caption('ارفع تسجيلًا لتفعيل زر التحليل.')
    else:
        st.caption('هذا الخيار يستخدم النموذج العربي الرسمي ArA-DF-Baseline ويحتاج ملفات كبيرة وبيئة DeepFense.')
        model_dir = st.text_input('مجلد النموذج على الخادم', value=os.getenv('MIYAR_MODEL_DIR', 'models/ArA-DF-Baseline'))
        required = [Path(model_dir) / name for name in ('best_model.pth', 'xlsr2_300m.pt', 'config.yaml', 'run_inference.py')]
        available = all(path.is_file() for path in required)
        if not available:
            st.warning('لم تثبت أوزان النموذج العربي الكبير. اختر النموذج السحابي الخفيف بالأعلى للتجربة.')
        if upload and st.button('تحليل بالنموذج العربي الكبير', disabled=not available):
            data = upload.getvalue()
            if len(data) > 25 * 1024 * 1024:
                st.error('الملف يتجاوز 25 ميجابايت')
            else:
                with tempfile.TemporaryDirectory() as td:
                    ext = '.wav' if upload.name.lower().endswith('.wav') else '.flac'
                    path = Path(td) / ('uploaded' + ext)
                    path.write_bytes(data)
                    try:
                        rp = None
                        if report_upload:
                            rp = Path(td) / 'threshold.json'
                            rp.write_bytes(report_upload.getvalue())
                        with st.spinner('تشغيل النموذج العربي...'):
                            result = infer_real_audio(path, model_dir, decision_report=rp)
                        st.json(result)
                    except Exception as exc:
                        st.error(f'تعذر تشغيل النموذج العربي: {exc}')



with three:
    st.subheader("فحص التلاعب الجزئي في التسجيلات العربية — نسخة تجريبية")
    st.info("مثال: استبدال «أربعة آلاف» بعبارة «أربعين ألفًا» باستخدام استنساخ الصوت. قد يكون بقية التسجيل أصليًا؛ لذلك فحص التسجيل كاملًا بمتوسط الدرجات قد يفوّت الجزء المعدل.")
    st.warning("النتائج هنا مؤشرات تجريبية وليست دليلًا جنائيًا أو تأكيدًا على استخدام الذكاء الاصطناعي. لم تُقَس بعد دقة تحديد كلمة عربية مولّدة أو معدل الإنذارات الخاطئة.")

    reference_mode, solo_mode = st.tabs(["مقارنة التسجيل المشبوه بالأصلي", "مسح تسجيل واحد دون الأصل"])

    with reference_mode:
        st.markdown("#### عند توفر التسجيل الأصلي الموثوق")
        st.caption("ارفع التسجيل الأول كما وصل من صاحبه، ثم النسخة محل الشك. نقارن البصمة الصوتية عبر الزمن، مع محاذاة التسجيلين. يمكن أن تظهر فروق بسبب الضغط أو التحرير العادي، وليس الذكاء الاصطناعي فقط.")
        original_file = st.file_uploader("التسجيل الأصلي الموثوق (WAV أو FLAC)", type=["wav","flac"], key="partial_original")
        suspect_file = st.file_uploader("التسجيل المشتبه به (WAV أو FLAC)", type=["wav","flac"], key="partial_suspect")
        if original_file and suspect_file:
            col_orig, col_susp = st.columns(2)
            with col_orig:
                st.caption("الأصلي")
                st.audio(original_file.getvalue())
            with col_susp:
                st.caption("المشتبه به")
                st.audio(suspect_file.getvalue())
            if st.button("مقارنة التسجيلين وتحديد الاختلافات", key="compare_audio", type="primary"):
                if max(len(original_file.getvalue()), len(suspect_file.getvalue())) > 25 * 1024 * 1024:
                    st.error("يجب ألا يزيد حجم أي ملف عن 25MB")
                else:
                    try:
                        with tempfile.TemporaryDirectory() as td:
                            a=Path(td) / ("original.flac" if original_file.name.lower().endswith(".flac") else "original.wav")
                            b=Path(td) / ("suspicious.flac" if suspect_file.name.lower().endswith(".flac") else "suspicious.wav")
                            a.write_bytes(original_file.getvalue())
                            b.write_bytes(suspect_file.getvalue())
                            with st.spinner("محاذاة التسجيلين والبحث عن فروق محلية..."):
                                result=compare_to_reference(a,b)
                        st.success("انتهت المقارنة الصوتية الاستكشافية")
                        st.metric("فترات اختلاف ملحوظة", len(result["regions"]))
                        if result["regions"]:
                            st.dataframe(pd.DataFrame(result["regions"]).rename(columns={
                                "start_seconds":"من الثانية","end_seconds":"إلى الثانية",
                                "peak_difference":"أعلى مؤشر اختلاف"}),use_container_width=True,hide_index=True)
                        else:
                            st.info("لم يُظهر مقياس المقارنة فروقًا موضعية بارزة. هذا لا يثبت أن الملف لم يُعدَّل.")
                        timeline=pd.DataFrame(result["timeline"]).set_index("time_seconds")
                        st.line_chart(timeline)
                        st.caption("الأفقي: زمن الملف المشتبه به بالثواني. الرأسي: مؤشر اختلاف صوتي نسبي، وليس احتمال تزييف.")
                        st.warning(result["warning"])
                        st.download_button("تنزيل تقرير مقارنة JSON",data=json.dumps(result,ensure_ascii=False,indent=2),
                                           file_name="miyar_partial_reference_comparison.json",mime="application/json")
                    except Exception as exc:
                        st.error(f"تعذرت المقارنة: {exc}")
        else:
            st.caption("يلزم تسجيلان للقيام بالمقارنة: نسخة أصلية موثوقة ونسخة مشتبه بها.")

    with solo_mode:
        st.markdown("#### عند توفر تسجيل مشتبه به فقط")
        st.caption("يُجرى فحص متداخل بنوافذ طولها 4 ثوانٍ؛ بدلاً من احتساب متوسط عدد محدود من النوافذ. تحديد كلمة قصيرة مثل «أربعين ألفًا» غير مضمون.")
        solo_file=st.file_uploader("التسجيل المراد مسحه (WAV أو FLAC)",type=["wav","flac"],key="partial_solo")
        step=st.select_slider("خطوة بدء النوافذ (ثوانٍ)",options=[1.0,2.0,3.0,4.0],value=2.0,
                              help="خطوة أصغر تعني نوافذ أكثر واستهلاك CPU أعلى؛ طول كل نافذة يبقى 4 ثوانٍ.")
        if solo_file:
            st.audio(solo_file.getvalue())
            if st.button("مسح مواضع الاشتباه في التسجيل",key="run_partial_scan",type="primary"):
                if len(solo_file.getvalue())>25*1024*1024:
                    st.error("الملف يتجاوز 25MB")
                else:
                    try:
                        with tempfile.TemporaryDirectory() as td:
                            ext=".flac" if solo_file.name.lower().endswith(".flac") else ".wav"
                            path=Path(td)/("scan"+ext)
                            path.write_bytes(solo_file.getvalue())
                            with st.spinner("تشغيل النموذج على النوافذ المتداخلة..."):
                                result=scan_partial_windows(path,cached_light_model(),step_seconds=step)
                        frames=pd.DataFrame(result["windows"])
                        st.line_chart(frames.set_index("start_seconds")["spoof_score"])
                        strongest=max(result["windows"],key=lambda x:x["spoof_score"])
                        st.write(f"**النافذة ذات أعلى درجة:** من {strongest['start_seconds']} إلى {strongest['end_seconds']} ثانية، بدرجة خام {strongest['spoof_score']:.4f}")
                        st.dataframe(frames,use_container_width=True,hide_index=True)
                        st.warning(result["warning"])
                        st.download_button("تنزيل تقرير النوافذ JSON",data=json.dumps(result,ensure_ascii=False,indent=2),
                                           file_name="miyar_partial_window_scan.json",mime="application/json")
                    except Exception as exc:
                        st.error(f"تعذر الفحص الموضعي: {exc}")
                        st.caption("قد تحتاج إلى تسجيل أقصر أو خطوة أكبر لتقليل عدد النوافذ وموارد الخادم.")

with four:
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
