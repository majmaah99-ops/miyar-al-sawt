# تشغيل مِعيار الصوت من GitHub

## تشغيل محلي
```bash
git clone https://github.com/majmaah99-ops/miyar-al-sawt.git
cd miyar-al-sawt
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run dashboard.py
```

على Windows يمكنك تشغيل `run_windows.bat` بعد تثبيت Python.

## تجربة لوحة العتبات
افتح `http://localhost:8501`، ثم تبويب **تدقيق العتبات**، وارفع الملف `examples/DEMO_SYNTHETIC_scores.csv`.

**الملف تعليمي اصطناعي؛ لا يمثل نتائج كشف حقيقية.**

## نشر الواجهة عبر Streamlit Community Cloud
- اربط حساب GitHub بالخدمة من https://share.streamlit.io/
- اختر المستودع `majmaah99-ops/miyar-al-sawt`، والفرع `main`، وملف التشغيل `dashboard.py`.
- انشر التطبيق ثم جرّب لوحة العتبات.
- فحص الصوت الحقيقي لن يعمل حتى تثبيت أوزان DeepFense والنموذج العربي في بيئة استضافة ملائمة.

## أمان
لا تُدرج تسجيلات خاصة أو أوزان نماذج أو مفاتيح سرية داخل المستودع.
