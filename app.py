
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

def find_sr_zones(df, lookback=260, max_each=3, min_gap_pct=0.02):
    """
    Classical S/R engine:
    - confirmed swing highs/lows
    - clusters nearby pivots into zones
    - a retest counts only after price clearly leaves and comes back
    - rewards rejection quality, volume, recency and role-reversal
    - penalizes repeated hammering, clean breaks and stale levels
    """
    x = df.tail(min(lookback, len(df))).copy().reset_index(drop=True)
    if len(x) < 60:
        return [], []

    current = float(x.close.iloc[-1])
    atr_s = atr(x, 14).replace([np.inf, -np.inf], np.nan)
    atr_now = float(atr_s.iloc[-1]) if pd.notna(atr_s.iloc[-1]) else current * 0.02
    if atr_now <= 0:
        atr_now = current * 0.02

    # Zone width: volatility-aware, but not so wide that it becomes useless.
    half_width = np.clip(atr_now * 0.24, current * 0.0035, current * 0.012)

    vol_ma = x.volume.rolling(20, min_periods=5).mean().replace(0, np.nan)
    pivots = []
    wing = 4  # 9-candle confirmed swing: filters minor noise

    for i in range(wing, len(x) - wing):
        lo = float(x.low.iloc[i]); hi = float(x.high.iloc[i])
        local_lows = x.low.iloc[i-wing:i+wing+1]
        local_highs = x.high.iloc[i-wing:i+wing+1]
        a = float(atr_s.iloc[i]) if pd.notna(atr_s.iloc[i]) else atr_now

        # Require some prominence versus nearby bars, not merely equal local extreme.
        low_prom = float(local_lows.drop(index=i).min()) - lo
        high_prom = hi - float(local_highs.drop(index=i).max())

        if lo <= float(local_lows.min()) and low_prom >= -0.12 * a:
            pivots.append({"price": lo, "idx": i, "kind": "L"})
        if hi >= float(local_highs.max()) and high_prom >= -0.12 * a:
            pivots.append({"price": hi, "idx": i, "kind": "H"})

    if not pivots:
        return [], []

    # Cluster pivots by ATR/price distance.
    pivots.sort(key=lambda p: p["price"])
    merge_dist = max(half_width * 1.55, current * 0.006)
    clusters = []
    for p in pivots:
        matches = [c for c in clusters if abs(p["price"] - c["center"]) <= merge_dist]
        if not matches:
            clusters.append({"members": [p], "center": p["price"]})
        else:
            c = min(matches, key=lambda q: abs(p["price"] - q["center"]))
            c["members"].append(p)
            c["center"] = float(np.median([m["price"] for m in c["members"]]))

    zones = []
    n = len(x)

    for c in clusters:
        center = float(c["center"])
        low, high = center - half_width, center + half_width
        leave_buffer = half_width + max(atr_now * 0.45, current * 0.005)

        # Genuine visits: once inside, price must leave decisively before another visit counts.
        visits = []
        armed = True
        for i in range(n):
            inside = float(x.low.iloc[i]) <= high and float(x.high.iloc[i]) >= low
            if inside and armed:
                visits.append(i)
                armed = False
            elif not armed:
                clearly_above = float(x.low.iloc[i]) > center + leave_buffer
                clearly_below = float(x.high.iloc[i]) < center - leave_buffer
                if clearly_above or clearly_below:
                    armed = True

        if len(visits) < 2:
            continue

        # Compress visits that are too close in time: same battle != many independent tests.
        independent = []
        for i in visits:
            if not independent or i - independent[-1] >= 6:
                independent.append(i)
        visits = independent
        if len(visits) < 2:
            continue

        reactions, vol_confirm, clean_breaks, side_history = [], [], 0, []
        for i in visits:
            a = float(atr_s.iloc[i]) if pd.notna(atr_s.iloc[i]) else atr_now
            before = float(x.close.iloc[max(0, i-1)])
            side = "above" if before > center else "below"
            side_history.append(side)

            future = x.iloc[i+1:min(n, i+7)]
            if future.empty:
                continue

            # Best rejection away from zone over the next six candles.
            up_move = max(0.0, float(future.high.max()) - high)
            down_move = max(0.0, low - float(future.low.min()))
            if side == "above":       # support test
                move = up_move
            else:                     # resistance test
                move = down_move
            reactions.append(move / max(a, 1e-9))

            base_vol = float(vol_ma.iloc[i]) if pd.notna(vol_ma.iloc[i]) else np.nan
            vr = float(x.volume.iloc[i]) / base_vol if np.isfinite(base_vol) and base_vol > 0 else 1.0
            vol_confirm.append(vr)

            # A decisive close through the zone with range expansion weakens it.
            candle_range = float(x.high.iloc[i] - x.low.iloc[i])
            if side == "above" and float(x.close.iloc[i]) < low - 0.20*a and candle_range > 1.05*a:
                clean_breaks += 1
            if side == "below" and float(x.close.iloc[i]) > high + 0.20*a and candle_range > 1.05*a:
                clean_breaks += 1

        if not reactions:
            continue

        reaction_med = float(np.median(reactions))
        strong_rejections = sum(r >= 0.8 for r in reactions)
        vol_med = float(np.median(vol_confirm)) if vol_confirm else 1.0

        # Classical role reversal: same zone acted from both sides at different times.
        role_reversal = len(set(side_history)) > 1

        # Recency decays smoothly; old levels can survive if reactions were exceptional.
        last_visit = max(visits)
        age = (n - 1 - last_visit)
        recency = float(np.exp(-age / 85.0))

        # Repeated tests are useful initially, then become "hammering" and weaken the level.
        test_score = {2: 2.2, 3: 3.2, 4: 3.7}.get(len(visits), 3.7)
        hammer_penalty = max(0, len(visits) - 4) * 0.75

        raw = (
            test_score
            + min(reaction_med, 2.5) * 2.15
            + min(strong_rejections, 3) * 0.65
            + np.clip(vol_med - 0.8, 0, 1.2) * 1.10
            + recency * 1.65
            + (1.0 if role_reversal else 0.0)
            - clean_breaks * 1.35
            - hammer_penalty
        )

        # Reject weak "levels" even if they accumulated touches.
        if reaction_med < 0.55 or raw < 6.0:
            continue

        if raw >= 10.5:
            strength, grade = 5, "قوية جدًا"
        elif raw >= 8.7:
            strength, grade = 4, "قوية"
        elif raw >= 7.2:
            strength, grade = 3, "جيدة"
        else:
            strength, grade = 2, "متوسطة"

        zones.append({
            "center": center, "low": low, "high": high,
            "touches": len(visits), "strength": strength, "grade": grade,
            "reaction": reaction_med, "volume_confirm": vol_med,
            "role_reversal": role_reversal, "breaks": clean_breaks,
            "recency": recency, "raw": float(raw)
        })

    # A zone may be support or resistance according to CURRENT price.
    supports = [z for z in zones if z["high"] < current]
    resistances = [z for z in zones if z["low"] > current]

    # Same-side separation: don't show several versions of essentially one level.
    def dedupe(items):
        items = sorted(items, key=lambda z: z["raw"], reverse=True)
        kept = []
        for z in items:
            sep = max(0.018, min_gap_pct * 0.85)
            if all(abs(z["center"] - k["center"]) / current >= sep for k in kept):
                kept.append(z)
        return kept

    supports, resistances = dedupe(supports), dedupe(resistances)

    # Relevance = quality with a modest distance penalty; avoids ancient faraway levels dominating.
    for z in supports + resistances:
        dist = abs(z["center"] - current) / current
        z["rank_score"] = z["raw"] - max(0.0, dist - 0.10) * 10.0

    # Enforce meaningful open space between nearest support and resistance.
    while supports and resistances:
        s = max(supports, key=lambda z: z["center"])
        r = min(resistances, key=lambda z: z["center"])
        gap = (r["low"] - s["high"]) / max((r["low"] + s["high"]) / 2, 1e-9)
        if gap >= min_gap_pct:
            break
        if s["rank_score"] <= r["rank_score"]:
            supports.remove(s)
        else:
            resistances.remove(r)

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
    sr=0.0
    if supports:
        s=supports[0]
        dist=max(0.0, (p-s["high"])/p)
        quality=s["strength"] + (0.75 if s["role_reversal"] else 0)
        if dist <= 0.015: sr += min(9.0, quality*1.55)
        elif dist <= 0.04: sr += min(5.0, quality)
    if resistances:
        r=resistances[0]
        dist=max(0.0, (r["low"]-p)/p)
        quality=r["strength"] + (0.75 if r["role_reversal"] else 0)
        if dist <= 0.015: sr -= min(9.0, quality*1.55)
        elif dist <= 0.04: sr -= min(5.0, quality)
    sr=int(round(np.clip(sr,-12,12)))
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
tf=c2.selectbox("الفريم الرئيسي",list(INTERVALS.keys()),index=3)

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

        def zone_line(i, zz, side):
            dist = ((price - zz["high"]) / price * 100) if side == "S" else ((zz["low"] - price) / price * 100)
            rr = " • تبادل أدوار" if zz["role_reversal"] else ""
            return (
                f"**{i}. {zone_text(zz)}** — {zz['grade']}  "
                f"({'★'*zz['strength']}{'☆'*(5-zz['strength'])})  \\n"
                f"اختبارات مستقلة: {zz['touches']} • ارتداد: {zz['reaction']:.1f} ATR "
                f"• بُعد: {max(0,dist):.1f}%{rr}"
            )

        with left:
            st.markdown("**🟢 الدعوم**")
            if supports:
                for i,zz in enumerate(supports,1):
                    st.markdown(zone_line(i,zz,"S"))
            else:
                st.write("لا يوجد دعم كلاسيكي موثوق قريب ضمن البيانات الحالية.")

        with right:
            st.markdown("**🔴 المقاومات**")
            if resistances:
                for i,zz in enumerate(resistances,1):
                    st.markdown(zone_line(i,zz,"R"))
            else:
                st.write("لا توجد مقاومة كلاسيكية موثوقة قريبة ضمن البيانات الحالية.")

        g,h=st.columns(2)
        g.metric("RSI",f"{rv:.1f}")
        h.metric("Relative Volume",f"{rvol:.2f}×")

        with st.expander("تفاصيل النتيجة"):
            st.write({k:f"{v:+.0f}" for k,v in details.items()})
            st.caption(f"ATR = {atrpct:.2f}% من السعر | آخر شمعة: {stamp} بتوقيت نيويورك")
            st.caption("Score يقيس اتفاق الإشارات الفنية، وليس نسبة احتمال مؤكدة.")

        st.info("المستويات هنا مناطق كلاسيكية وليست نقاطًا لحظية: تُبنى من Swing High/Low مؤكدة، ويُحسب الاختبار مرة جديدة فقط بعد ابتعاد السعر وعودته. القوة تعتمد على جودة الرفض السعري، استقلال الاختبارات، الحجم، الحداثة، تبادل الأدوار والكسر النظيف. كثرة اللمسات بعد حد معيّن تُضعف المستوى بدل أن ترفعه، وتُفلتر المناطق المتقاربة مع حد أدنى مستهدف 2% بين أقرب دعم ومقاومة.")
    except Exception as e:
        safe=str(e)
        if API_KEY: safe=safe.replace(API_KEY,"***")
        st.error(f"تعذر التحليل: {safe}")
