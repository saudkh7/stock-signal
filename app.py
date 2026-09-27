
import os, math
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Signal", page_icon="📈", layout="wide")

API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", os.getenv("TWELVE_DATA_API_KEY", ""))

INTERVALS = {"15m":"15min","1H":"1h","4H":"4h","1D":"1day"}

def fetch_series(symbol, interval, outputsize=500):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol.upper().strip(),
        "interval": interval,
        "outputsize": outputsize,
        "order": "ASC",
        "timezone": "America/New_York",
        "apikey": API_KEY,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    if "values" not in j:
        raise RuntimeError(j.get("message", str(j)))
    df = pd.DataFrame(j["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([(df.high-df.low), (df.high-pc).abs(), (df.low-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def enrich(df):
    x = df.copy()
    for n in (20,50,200): x[f"ema{n}"] = ema(x.close,n)
    x["rsi"] = rsi(x.close)
    e12, e26 = ema(x.close,12), ema(x.close,26)
    x["macd"] = e12-e26
    x["macd_signal"] = ema(x.macd,9)
    x["macd_hist"] = x.macd-x.macd_signal
    x["atr"] = atr(x)
    x["vol_ma20"] = x.volume.rolling(20).mean()
    x["rvol"] = x.volume / x.vol_ma20.replace(0,np.nan)
    return x

def support_resistance(df, lookback=80):
    x = df.tail(lookback)
    p = float(x.close.iloc[-1])
    # confirmed local pivots
    lows, highs = [], []
    for i in range(2, len(x)-2):
        if x.low.iloc[i] <= x.low.iloc[i-2:i+3].min(): lows.append(float(x.low.iloc[i]))
        if x.high.iloc[i] >= x.high.iloc[i-2:i+3].max(): highs.append(float(x.high.iloc[i]))
    supports = [v for v in lows if v < p]
    resistances = [v for v in highs if v > p]
    sup = max(supports) if supports else float(x.low.min())
    res = min(resistances) if resistances else float(x.high.max())
    return sup, res

def score_frame(df):
    x = enrich(df)
    z = x.iloc[-1]
    prev = x.iloc[-2]
    score = 0.0
    details = {}

    # Trend: 30
    trend = 0
    trend += 10 if z.close > z.ema20 else -10
    trend += 10 if z.ema20 > z.ema50 else -10
    trend += 10 if z.ema50 > z.ema200 else -10
    score += trend; details["Trend"] = trend

    # MACD: 20
    m = 0
    m += 10 if z.macd > z.macd_signal else -10
    m += 5 if z.macd > 0 else -5
    m += 5 if z.macd_hist > prev.macd_hist else -5
    score += m; details["MACD"] = m

    # RSI: 15 (momentum, penalize extremes)
    rv = float(z.rsi)
    if 55 <= rv <= 70: rs = 15
    elif 50 <= rv < 55: rs = 7
    elif 30 <= rv < 45: rs = -15
    elif 45 <= rv < 50: rs = -7
    elif rv > 75: rs = -5
    elif rv < 25: rs = 5
    else: rs = 0
    score += rs; details["RSI"] = rs

    # Volume confirmation: 15
    direction = 1 if z.close > prev.close else -1
    rvol = 0 if pd.isna(z.rvol) else float(z.rvol)
    if rvol >= 1.5: vs = 15*direction
    elif rvol >= 1.1: vs = 9*direction
    elif rvol >= .8: vs = 4*direction
    else: vs = 0
    score += vs; details["Volume"] = vs

    # S/R positioning: 15
    sup,res = support_resistance(x)
    span = max(res-sup, 1e-9)
    pos = (float(z.close)-sup)/span
    sr = 15 if pos > .75 and z.close > prev.close else (-15 if pos < .25 and z.close < prev.close else (5 if pos >= .5 else -5))
    score += sr; details["S/R"] = sr

    # ATR: confidence modifier only, not direction
    atr_pct = float(z.atr/z.close*100) if z.close else 0
    score = max(-100, min(100, score))
    return round(score), details, sup, res, rv, rvol, atr_pct, x

def label(score):
    if score >= 60: return "صاعد قوي"
    if score >= 25: return "ميل صاعد"
    if score > -25: return "محايد"
    if score > -60: return "ميل هابط"
    return "هابط قوي"

st.title("Stock Signal")
st.caption("واجهة نظيفة — المؤشرات تُحسب في الخلفية ولا تملأ الشارت.")

if not API_KEY:
    st.error("أضف مفتاح Twelve Data المجاني باسم TWELVE_DATA_API_KEY في Secrets أو متغيرات البيئة.")
    st.stop()

c1,c2 = st.columns([2,1])
symbol = c1.text_input("رمز السهم", "CRWV").upper().strip()
tf = c2.selectbox("الفريم الرئيسي", list(INTERVALS.keys()), index=1)

if st.button("تحليل", type="primary", use_container_width=True):
    try:
        with st.spinner("جاري جلب البيانات وحساب الإشارات..."):
            results={}
            for name,interval in INTERVALS.items():
                d=fetch_series(symbol, interval)
                results[name]=score_frame(d)

        main=results[tf]
        score,details,sup,res,rv,rvol,atrpct,x=main
        price=float(x.close.iloc[-1])
        stamp=x.datetime.iloc[-1]

        a,b,c,d=st.columns(4)
        a.metric(symbol, f"${price:,.2f}")
        b.metric("Score", f"{score:+d}/100")
        c.metric("الاتجاه", label(score))
        aligned=sum(1 for v in results.values() if (v[0]>24)==(score>24) and (v[0]<-24)==(score<-24))
        conf = "مرتفعة" if aligned>=3 and abs(score)>=45 else ("متوسطة" if aligned>=2 else "منخفضة")
        d.metric("الثقة", conf)

        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.82,0.18],vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=x.datetime,open=x.open,high=x.high,low=x.low,close=x.close,name="Price"),row=1,col=1)
        fig.add_trace(go.Bar(x=x.datetime,y=x.volume,name="Volume"),row=2,col=1)
        fig.update_layout(height=560,margin=dict(l=10,r=10,t=20,b=10),xaxis_rangeslider_visible=False,showlegend=False)
        st.plotly_chart(fig,use_container_width=True)

        st.subheader("الفريمات")
        cols=st.columns(4)
        for col,(name,val) in zip(cols,results.items()):
            col.metric(name, f"{val[0]:+d}", label(val[0]))

        e,f,g,h=st.columns(4)
        e.metric("الدعم",f"${sup:,.2f}")
        f.metric("المقاومة",f"${res:,.2f}")
        g.metric("RSI",f"{rv:.1f}")
        h.metric("Relative Volume",f"{rvol:.2f}×")

        with st.expander("لماذا هذه النتيجة؟"):
            st.write({k:f"{v:+.0f}" for k,v in details.items()})
            st.caption(f"ATR = {atrpct:.2f}% من السعر. آخر شمعة مستلمة: {stamp} بتوقيت نيويورك.")
            st.caption("Score هو مقياس ميل فني، وليس احتمالًا إحصائيًا للصعود. نحتاج Backtest قبل تحويله إلى نسب نجاح.")
    except Exception as e:
        st.error(f"تعذر التحليل: {e}")
