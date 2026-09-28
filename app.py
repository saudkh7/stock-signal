
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

def find_sr_zones(df, lookback=220, max_each=3, min_gap_pct=0.02):
    """
    Build meaningful support/resistance zones from confirmed pivots.
    A retest counts only after price leaves the zone and later returns.
    Micro-levels are filtered; nearest opposite zones must be >= min_gap_pct apart.
    """
    x = df.tail(min(lookback, len(df))).copy().reset_index(drop=True)
    if len(x) < 40:
        return [], []

    current = float(x.close.iloc[-1])
    atr_series = atr(x, 14)
    atr_now = float(atr_series.iloc[-1])
    if not np.isfinite(atr_now) or atr_now <= 0:
        atr_now = current * 0.02

    # Zone width adapts to volatility, but remains a zone rather than a single tick.
    half_width = max(current * 0.004, atr_now * 0.28)
    half_width = min(half_width, current * 0.018)

    pivots = []
    # Wider 7-bar pivot confirmation removes a lot of intraday/noisy swings.
    for i in range(3, len(x)-3):
        lo = float(x.low.iloc[i]); hi = float(x.high.iloc[i])
        local_low = float(x.low.iloc[i-3:i+4].min())
        local_high = float(x.high.iloc[i-3:i+4].max())
        if lo <= local_low:
            pivots.append({"price":lo, "idx":i, "kind":"L", "vol":float(x.volume.iloc[i] or 0)})
        if hi >= local_high:
            pivots.append({"price":hi, "idx":i, "kind":"H", "vol":float(x.volume.iloc[i] or 0)})

    if not pivots:
        return [], []

    # Price clustering.
    pivots.sort(key=lambda q: q["price"])
    clusters = []
    merge_dist = max(half_width * 1.75, current * 0.0075)
    for q in pivots:
        candidates = [c for c in clusters if abs(q["price"]-c["center"]) <= merge_dist]
        if not candidates:
            clusters.append({"prices":[q["price"]], "idxs":[q["idx"]], "vols":[q["vol"]],
                             "kinds":[q["kind"]], "center":q["price"]})
        else:
            c = min(candidates, key=lambda z: abs(q["price"]-z["center"]))
            c["prices"].append(q["price"]); c["idxs"].append(q["idx"])
            c["vols"].append(q["vol"]); c["kinds"].append(q["kind"])
            c["center"] = float(np.median(c["prices"]))

    avg_vol = float(x.volume.replace(0, np.nan).mean())
    if not np.isfinite(avg_vol) or avg_vol <= 0:
        avg_vol = 1.0

    ranked = []
    n = len(x)
    for c in clusters:
        center = c["center"]
        low, high = center-half_width, center+half_width

        # True tests: price must leave by a buffer before a new return is counted.
        tests = []
        armed = True
        leave_buffer = half_width + max(atr_now * 0.35, current * 0.004)
        for i in range(n):
            intersects = float(x.low.iloc[i]) <= high and float(x.high.iloc[i]) >= low
            if intersects and armed:
                tests.append(i)
                armed = False
            elif not armed:
                far = (float(x.low.iloc[i]) > center + leave_buffer or
                       float(x.high.iloc[i]) < center - leave_buffer)
                if far:
                    armed = True

        # Reaction after pivot: move away within 5 candles, normalized by ATR at that time.
        reactions = []
        for i in c["idxs"]:
            if i+5 < n:
                local_atr = float(atr_series.iloc[i])
                if not np.isfinite(local_atr) or local_atr <= 0:
                    local_atr = atr_now
                if c["kinds"][c["idxs"].index(i)] == "L":
                    move = float(x.high.iloc[i+1:i+6].max()) - center
                else:
                    move = center - float(x.low.iloc[i+1:i+6].min())
                reactions.append(max(0.0, move) / max(local_atr, 1e-9))

        reaction = float(np.mean(reactions)) if reactions else 0.0
        recency = max(c["idxs"]) / max(n-1, 1)
        vol_ratio = float(np.mean(c["vols"]) / avg_vol) if c["vols"] else 1.0
        true_tests = len(tests)

        # Require evidence: at least 2 genuine tests, or a very strong repeated pivot cluster.
        if true_tests < 2 and len(c["idxs"]) < 3:
            continue
        if reaction < 0.55:
            continue

        raw = (
            min(true_tests, 4) * 2.0 +
            min(len(c["idxs"]), 5) * 0.65 +
            min(reaction, 3.0) * 1.15 +
            min(vol_ratio, 2.0) * 0.65 +
            recency * 1.15
        )
        strength = int(np.clip(round(raw / 2.6), 1, 5))
        ranked.append({
            "center":center, "low":low, "high":high,
            "touches":true_tests, "strength":strength,
            "reaction":reaction, "raw":raw
        })

    supports = [z for z in ranked if z["high"] < current]
    resistances = [z for z in ranked if z["low"] > current]

    # Remove nearly duplicate zones within the same side, keeping the stronger one.
    def dedupe(items):
        items = sorted(items, key=lambda z: z["raw"], reverse=True)
        kept = []
        for z in items:
            if all(abs(z["center"]-k["center"]) / current >= 0.018 for k in kept):
                kept.append(z)
        return kept

    supports = dedupe(supports)
    resistances = dedupe(resistances)

    # Prefer relevant zones: strength first, but penalize extremely distant history.
    for z in supports + resistances:
        dist = abs(z["center"]-current) / current
        z["rank_score"] = z["raw"] - max(0, dist-0.12)*8

    supports = sorted(supports, key=lambda z: z["rank_score"], reverse=True)
    resistances = sorted(resistances, key=lambda z: z["rank_score"], reverse=True)

    # Ensure the nearest displayed support/resistance are meaningfully separated.
    # We target at least 2%; if a pair is too tight, drop the weaker micro-zone.
    changed = True
    while supports and resistances and changed:
        changed = False
        s = max(supports, key=lambda z: z["center"])
        r = min(resistances, key=lambda z: z["center"])
        gap = (r["low"] - s["high"]) / max((r["low"] + s["high"])/2, 1e-9)
        if gap < min_gap_pct:
            if s["raw"] <= r["raw"]:
                supports.remove(s)
            else:
                resistances.remove(r)
            changed = True

    # Final display: nearest meaningful zones first.
    supports = sorted(supports, key=lambda z: z["center"], reverse=True)[:max_each]
    resistances = sorted(resistances, key=lambda z: z["center"])[:max_each]
    return supports, resistances

def zone_text(z):
    return f"${z['low']:.2f}–${z['high']:.2f}"


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
    supports,resistances=find_sr_zones(x, min_gap_pct=0.02)
    p=float(z.close); atr_now=max(float(z.atr), p*0.005)
    sr=0
    if supports:
        s=supports[0]
        dist=max(0.0, (p-s["high"])/p)
        if dist <= 0.012: sr += min(10, 2*s["strength"])
        elif dist <= 0.03: sr += min(6, s["strength"])
    if resistances:
        r=resistances[0]
        dist=max(0.0, (r["low"]-p)/p)
        if dist <= 0.012: sr -= min(10, 2*r["strength"])
        elif dist <= 0.03: sr -= min(6, r["strength"])
    sr=int(np.clip(sr,-15,15))
    score+=sr; details["S/R"]=sr
    atrpct=float(z.atr/z.close*100) if z.close else 0
    return round(max(-100,min(100,score))),details,supports,resistances,rv,rvol,atrpct,x

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

        score,details,supports,resistances,rv,rvol,atrpct,x=main
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

        # Shade only the nearest/highest-ranked S/R zones to keep chart clean.
        for zz in supports:
            fig.add_hrect(y0=zz["low"], y1=zz["high"], opacity=0.10, line_width=0, row=1, col=1)
        for zz in resistances:
            fig.add_hrect(y0=zz["low"], y1=zz["high"], opacity=0.10, line_width=0, row=1, col=1)

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

        st.subheader("مناطق الدعم والمقاومة")
        left,right=st.columns(2)
        with left:
            st.markdown("**🟢 الدعوم**")
            if supports:
                for i,zz in enumerate(supports,1):
                    st.write(f"{i}. {zone_text(zz)}  |  {'★'*zz['strength']}{'☆'*(5-zz['strength'])}  |  اختبارات حقيقية: {zz['touches']}")
            else:
                st.write("لا توجد منطقة دعم موثوقة ضمن النطاق المحلل.")
        with right:
            st.markdown("**🔴 المقاومات**")
            if resistances:
                for i,zz in enumerate(resistances,1):
                    st.write(f"{i}. {zone_text(zz)}  |  {'★'*zz['strength']}{'☆'*(5-zz['strength'])}  |  اختبارات حقيقية: {zz['touches']}")
            else:
                st.write("لا توجد منطقة مقاومة موثوقة ضمن النطاق المحلل.")

        g,h=st.columns(2)
        g.metric("RSI",f"{rv:.1f}")
        h.metric("Relative Volume",f"{rvol:.2f}×")

        with st.expander("تفاصيل النتيجة"):
            st.write({k:f"{v:+.0f}" for k,v in details.items()})
            st.caption(f"ATR = {atrpct:.2f}% من السعر | آخر شمعة: {stamp} بتوقيت نيويورك")
            st.caption("Score يقيس اتفاق الإشارات الفنية، وليس نسبة احتمال مؤكدة.")

        st.info("المناطق المعروضة تُفلتر المستويات اللحظية: الاختبار الجديد لا يُحسب إلا بعد ابتعاد السعر ثم عودته، وتُراعى قوة الارتداد والحجم والحداثة وATR. كما تُستبعد المناطق المتقاربة جدًا، مع حد أدنى مستهدف 2% بين أقرب دعم ومقاومة.")
    except Exception as e:
        safe=str(e)
        if API_KEY: safe=safe.replace(API_KEY,"***")
        st.error(f"تعذر التحليل: {safe}")
