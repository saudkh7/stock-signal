
# Stock Signal — V1

نسخة أولى لأداة تحليل فني نظيفة للأسهم الأمريكية.

## ما تحسبه في الخلفية
- EMA 20 / 50 / 200
- RSI 14
- MACD 12/26/9
- Relative Volume
- ATR
- Support / Resistance
- Multi-timeframe: 15m, 1H, 4H, 1D

## العرض
الشارت يعرض السعر والحجم فقط. المؤشرات الأخرى لا تُرسم على الشارت.

## مصدر البيانات
Twelve Data. تحتاج مفتاح API مجاني.

## التشغيل
1. أنشئ حسابًا مجانيًا في Twelve Data وخذ API key.
2. ثبّت المتطلبات:
   `pip install -r requirements.txt`
3. عرّف المفتاح:
   - macOS/Linux: `export TWELVE_DATA_API_KEY="YOUR_KEY"`
   - Windows PowerShell: `$env:TWELVE_DATA_API_KEY="YOUR_KEY"`
4. شغّل:
   `streamlit run app.py`

يمكن أيضًا نشره على Streamlit Community Cloud وإضافة المفتاح في Secrets:
`TWELVE_DATA_API_KEY="YOUR_KEY"`

## ملاحظة مهمة
الـ Score من -100 إلى +100 يقيس اتفاق الإشارات الفنية، وليس "نسبة احتمال" حركة السعر.
الخطوة التالية المقترحة هي Backtest تاريخي لقياس نسبة نجاح كل نطاق Score ثم تعديل الأوزان.
