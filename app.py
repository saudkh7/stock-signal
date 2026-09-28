
import os, time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Signal", page_icon="📈", layout="wide")

API_KEY = st.secrets.get("TWELVE_DATA_API_KEY", os.getenv("TWELVE_DATA_API_KEY", ""))
INTERVALS = {"15m":"15min","1H":"1h","4H":"4h","1D":"1day"}

@st.cache_data(ttl=300, show_spinner=False)
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
    r = requests.get(url, params=params, timeout=25)
    if r.status_code == 429:
        raise RuntimeError("وصلنا لحد الطلبات المجانية مؤقتًا. انتظر حوالي دقيقة ثم حاول مرة أخرى.")
    if not r.ok:
        raise RuntimeError(f"تعذر جلب البيانات من مزود السوق (HTTP {r.status_code}).")
    j = r.json()
    if "values" not in j:
        msg = j.get("message", "لم تصل بيانات سعرية.")
        # Never print request URL / API key
        raise RuntimeError(str(msg).replace(API_KEY, "***") if API_KEY else str(msg))
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
    tr = pd.concat([(df.high-df.low),(df.high-pc).abs(),(df.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def enrich(df):
    x=df.copy()
    for n in (20,50,200): x[f"ema{n}"]=ema(x.close,n)
    x["rsi"]=rsi(x.close)
    e12,e26=ema(x.close,12),ema(x.close,26)
    x["macd"]=e12-e26
    x["macd_signal"]=ema(x.macd,9)
    x["macd_hist"]=x.macd-x.macd_signal
    x["atr"]=atr(x)
    x["vol_ma20"]=x.volume.rolling(20).mean()
    x["rvol"]=x.volume/x.vol_ma20.replace(0,np.nan)
    return x

def support_resistance(df, lookback=80):
    x=df.tail(lookback); p=float(x.close.iloc[-1])
    lows=[]; highs=[]
    for i in range(2,len(x)-2):
        if x.low.iloc[i] <= x.low.iloc[i-2:i+3].min(): lows.append(float(x.low.iloc[i]))
        if x.high.iloc[i] >= x.high.iloc[i-2:i+3].max(): highs.append(float(x.high.iloc[i]))
    supports=[v for v in lows if v<p]; resistances=[v for v in highs if v>p]
    return (max(supports) if supports else float(x.low.min()),
            min(resistances) if resistances else float(x.high.max()))

def score_frame(df):
    x=enrich(df); z=x.iloc[-1]; prev=x.iloc[-2]
    score=0.; details={}
    trend=(10 if z.close>z.ema20 else -10)+(10 if z.ema20>z.ema50 else -10)+(10 if z.ema50>z.ema200 else -10)
    score+=trend; details["Trend"]=trend
    m=(10 if z.macd>z.macd_signal else -10)+(5 if z.macd>0 else -5)+(5 if z.macd_hist>prev.macd_hist else -5)
    score+=m; details["MACD"]=m
    rv=float(z.rsi)
    if 55<=rv<=70: rs=15
    elif 50<=rv<55: rs=7
    elif 30<=rv<45: rs=-15
    elif 45<=rv<50: rs=-7
    elif rv>75: rs=-5
    elif rv<25: rs=5
    else: rs=0
    score+=rs; details["RSI"]=rs
    direction=1 if z.close>prev.close else -1
    rvol=0 if pd.isna(z.rvol) else float(z.rvol)
    vs=15*direction if rvol>=1.5 else (9*direction if rvol>=1.1 else (4*direction if rvol>=.8 else 0))
    score+=vs; details["Volume"]=vs
    sup,res=support_resistance(x); span=max(res-sup,1e-9); pos=(float(z.close)-sup)/span
    sr=15 if pos>.75 and z.close>prev.close else (-15 if pos<.25 and z.close<prev.close else (5 if pos>=.5 else -5))
    score+=sr; details["S/R"]=sr
    atrpct=float(z.atr/z.close*100) if z.close else 0
    return round(max(-100,min(100,score))),details,sup,res,rv,rvol,atrpct,x

def label(s):
    if s>=60:return "صاعد قوي"
    if s>=25:return "ميل صاعد"
    if s>-25:return "محايد"
    if s>-60:return "ميل هابط"
    return "هابط قوي"

st.title("Stock Signal")
st.caption("شموع + حجم تداول فقط على الشارت. التحليل الفني يُحسب في الخلفية.")

if not API_KEY:
    st.error("مفتاح البيانات غير موجود في Streamlit Secrets.")
    st.stop()

c1,c2=st.columns([2,1])
symbol=c1.text_input("رمز السهم","CRWV").upper().strip()
tf=c2.selectbox("الفريم الرئيسي",list(INTERVALS.keys()),index=1)

if st.button("تحليل",type="primary",use_container_width=True):
    try:
        with st.spinner("جاري جلب البيانات..."):
            d=fetch_series(symbol,INTERVALS[tf])
            main=score_frame(d)

        score,details,sup,res,rv,rvol,atrpct,x=main
        price=float(x.close.iloc[-1]); stamp=x.datetime.iloc[-1]

        a,b,c,dcol=st.columns(4)
        a.metric(symbol,f"${price:,.2f}")
        b.metric("Score",f"{score:+d}/100")
        c.metric("الاتجاه",label(score))
        dcol.metric("الفريم",tf)

        # Interactive candlestick chart with crosshair/spikes
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.82,0.18],vertical_spacing=0.025)
        fig.add_trace(go.Candlestick(
            x=x.datetime,open=x.open,high=x.high,low=x.low,close=x.close,
            name="Price",
            hovertext=[
                f"وقت: {dt}<br>فتح: ${o:.2f}<br>أعلى: ${h:.2f}<br>أدنى: ${l:.2f}<br>إغلاق: ${c:.2f}"
                for dt,o,h,l,c in zip(x.datetime,x.open,x.high,x.low,x.close)
            ],
            hoverinfo="text"
        ),row=1,col=1)
        fig.add_trace(go.Bar(x=x.datetime,y=x.volume,name="Volume",hovertemplate="الحجم: %{y:,.0f}<extra></extra>"),row=2,col=1)

        fig.update_layout(
            height=620, margin=dict(l=8,r=8,t=15,b=8),
            xaxis_rangeslider_visible=False, showlegend=False,
            hovermode="closest", dragmode="pan"
        )
        # Vertical + horizontal crosshair guides
        fig.update_xaxes(showspikes=True,spikesnap="cursor",spikemode="across",spikethickness=1)
        fig.update_yaxes(showspikes=True,spikesnap="cursor",spikemode="across",spikethickness=1,row=1,col=1)
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig,use_container_width=True,config={"scrollZoom":True,"displaylogo":False})

        e,f,g,h=st.columns(4)
        e.metric("الدعم",f"${sup:,.2f}")
        f.metric("المقاومة",f"${res:,.2f}")
        g.metric("RSI",f"{rv:.1f}")
        h.metric("Relative Volume",f"{rvol:.2f}×")

        with st.expander("تفاصيل النتيجة"):
            st.write({k:f"{v:+.0f}" for k,v in details.items()})
            st.caption(f"ATR = {atrpct:.2f}% من السعر | آخر شمعة: {stamp} بتوقيت نيويورك")
            st.caption("Score يقيس اتفاق الإشارات الفنية، وليس نسبة احتمال مؤكدة.")

        st.info("لتقليل أخطاء الخطة المجانية، النسخة الحالية تحلل الفريم المختار فقط. سنضيف مقارنة الفريمات بعد ضبط استهلاك البيانات.")
    except Exception as e:
        safe=str(e)
        if API_KEY: safe=safe.replace(API_KEY,"***")
        st.error(f"تعذر التحليل: {safe}")
