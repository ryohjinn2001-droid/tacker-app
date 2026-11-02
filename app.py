"""
Training & Conditioning Tracker v3
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import os

st.set_page_config(
    page_title="Training & Conditioning Tracker v3",
    page_icon="🥋",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_SESSIONS_FILE = "workout_sessions_v3.csv"
DATA_DAILY_FILE = "daily_condition_v3.csv"

DEFAULT_AGE = 25
DEFAULT_HR_MAX = 220 - DEFAULT_AGE  # 195 bpm

CATEGORY_DEFAULTS = {
    "ランニング（心拍計測: Banister法）": {"rpe": None, "desc": "平均心拍数・運動時間・安静時心拍数からBanister予備心拍TRIMPを自動計算"},
    "ランニング（ペース・環境係数法）": {"rpe": None, "desc": "資料準拠のペース係数×距離×路面×天候でTRIMPを算出"},
    "柔道（乱取・体力強化）": {"rpe": 8, "desc": "立技・寝技の乱取、サーキット、打込高強度 (RPE×時間)"},
    "柔道（技術練習・打込・型）": {"rpe": 4, "desc": "技の研究、基本打込、型練習、約束練習 (RPE×時間)"},
    "ウェイトトレーニング": {"rpe": 7, "desc": "スクワット、ベンチプレス、懸垂等の筋力・パワー系 (RPE×時間)"},
    "一般有酸素・その他運動": {"rpe": 5, "desc": "主観的強度 RPE×時間で算出"}
}

PACE_COEFFS = {
    "ジョグ (> 5'00/km)": 1.0,
    "ペース走 (4'59 - 4'00/km)": 1.5,
    "テンポ走 (3'59 - 3'41/km)": 3.0,
    "閾値走 (3'40 - 3'33/km)": 3.5,
    "インターバル (3'32 - 3'29/km)": 4.0,
    "インターバル (3'28 - 3'21/km)": 4.5,
    "インターバル (3'20 - 3'11/km)": 5.0,
    "インターバル (3'10 - 3'01/km)": 8.0,
    "スプリント走 (< 3'00/km)": 10.0
}

SURFACE_COEFFS = {
    "ロード (アスファルト)": 1.10,
    "トラック (オールウェザー)": 1.00,
    "グラウンド (土・不整地)": 1.25
}

WEATHER_COEFFS = {
    "晴れ・涼しい": 1.00,
    "晴れ・暑い": 1.30,
    "雨・涼しい": 1.10,
    "雨・暑い": 1.35
}

def init_sample_data():# init_sample_data()

    if not os.path.exists(DATA_DAILY_FILE):
        dates = pd.date_range(end=datetime.date.today(), periods=21)
        daily = []
        for d in dates:
            d_str = d.strftime("%Y-%m-%d")
            weekday = d.weekday()
            sleep_start = "21:30"
            sleep_end = "06:00" if weekday in [5, 6] else "05:30"
            weight = 75.0 + np.random.normal(0, 0.3)
            rhr = 52 + int(np.random.normal(0, 2))
            hrv = 68 + int(np.random.normal(0, 5))
            daily.append({
                "date": d_str,
                "sleep_start": sleep_start,
                "sleep_end": sleep_end,
                "weight": round(weight, 1),
                "rhr": rhr,
                "hrv": hrv
            })
        pd.DataFrame(daily).to_csv(DATA_DAILY_FILE, index=False)

    if not os.path.exists(DATA_SESSIONS_FILE):
        dates = pd.date_range(end=datetime.date.today(), periods=21)
        sessions = []
        for d in dates:
            d_str = d.strftime("%Y-%m-%d")
            weekday = d.weekday()
            if weekday in [0, 2, 4]:
                sessions.append({
                    "date": d_str, "category": "柔道（乱取・体力強化）",
                    "duration": 90, "rpe": 8, "load": 90 * 8, "calc_method": "Session-RPE", "note": "立技乱取4分x8本"
                })
                sessions.append({
                    "date": d_str, "category": "ウェイトトレーニング",
                    "duration": 45, "rpe": 7, "load": 45 * 7, "calc_method": "Session-RPE", "note": "スクワット・懸垂"
                })
            elif weekday in [1, 3]:
                sessions.append({
                    "date": d_str, "category": "柔道（技術練習・打込・型）",
                    "duration": 60, "rpe": 4, "load": 60 * 4, "calc_method": "Session-RPE", "note": "組手構造研究・移動打込"
                })
                sessions.append({
                    "date": d_str, "category": "ランニング（心拍計測: Banister法）",
                    "duration": 35, "rpe": 5, "load": 85.2, "calc_method": "Banister-HR", "note": "心拍走 (Avg:148bpm)"
                })
            elif weekday == 5:
                sessions.append({
                    "date": d_str, "category": "柔道（乱取・体力強化）",
                    "duration": 120, "rpe": 9, "load": 120 * 9, "calc_method": "Session-RPE", "note": "合同練習・試合形式"
                })
        pd.DataFrame(sessions).to_csv(DATA_SESSIONS_FILE, index=False)

init_sample_data()

def calc_sleep_hours(start_str, end_str):
    try:
        t_start = datetime.datetime.strptime(start_str, "%H:%M")
        t_end = datetime.datetime.strptime(end_str, "%H:%M")
        diff_minutes = (t_end - t_start).total_seconds() / 60
        if diff_minutes < 0:
            diff_minutes += 24 * 60
        return round(diff_minutes / 60, 2)
    except:
        return 0.0

def calc_banister_trimp(duration_min, hr_avg, hr_rest, hr_max=195, gender="male"):
    if hr_max <= hr_rest or hr_avg <= hr_rest:
        return 0.0
    delta_hr_ratio = (hr_avg - hr_rest) / (hr_max - hr_rest)
    delta_hr_ratio = max(0.0, min(1.0, delta_hr_ratio))
    b_val = 1.92 if gender == "male" else 1.67
    mult = np.exp(b_val * delta_hr_ratio)
    trimp = duration_min * delta_hr_ratio * mult
    return round(trimp, 1)

def compute_banister_model(daily_df, sessions_df, tau1=45, tau2=15, k1=1.0, k2=2.0):
    daily = daily_df.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    if not sessions_df.empty:
        sess = sessions_df.copy()
        sess["date"] = pd.to_datetime(sess["date"])
        trimp_sum = sess.groupby("date")["load"].sum().reset_index()
        trimp_sum.columns = ["date", "day_trimp"]
        daily = pd.merge(daily, trimp_sum, on="date", how="left").fillna({"day_trimp": 0})
    else:
        daily["day_trimp"] = 0

    daily["sleep_hours"] = daily.apply(lambda r: calc_sleep_hours(r["sleep_start"], r["sleep_end"]), axis=1)
    daily["sleep_7d_total"] = daily["sleep_hours"].rolling(window=7, min_periods=1).sum().round(1)
    daily["hrv_7d_avg"] = daily["hrv"].rolling(window=7, min_periods=1).mean().round(1)
    daily["rhr_mean"] = daily["rhr"].expanding(min_periods=1).mean().round(1)
    daily["prev_day_trimp"] = daily["day_trimp"].shift(1).fillna(0)

    n = len(daily)
    fitness = np.zeros(n)
    fatigue = np.zeros(n)

    decay_fit = np.exp(-1.0 / tau1)
    decay_fat = np.exp(-1.0 / tau2)

    for t in range(n):
        w_t = daily.loc[t, "day_trimp"]
        if t == 0:
            fitness[t] = w_t
            fatigue[t] = w_t
        else:
            fitness[t] = w_t + fitness[t - 1] * decay_fit
            fatigue[t] = w_t + fatigue[t - 1] * decay_fat

    daily["fitness"] = np.round(fitness, 1)
    daily["fatigue"] = np.round(fatigue, 1)
    daily["performance"] = np.round(k1 * daily["fitness"] - k2 * daily["fatigue"], 1)

    return daily

sessions_df = pd.read_csv(DATA_SESSIONS_FILE)
daily_df = pd.read_csv(DATA_DAILY_FILE)
avg_rhr_recorded = round(daily_df["rhr"].mean(), 1) if not daily_df.empty else 52.0

st.title("🥋 統合トレーニング & コンディション分析システム v3")
st.markdown("フィットネス-疲労理論 (Banister) とセッションRPE、心拍TRIMP、前日リカバリー（睡眠・RHR・HRV）一元管理ダッシュボード")

st.sidebar.header("📝 データ入力")
entry_type = st.sidebar.radio("入力項目を選択", ["練習セッション記録 (ランTRIMP対応)", "朝のコンディション記録 (前日睡眠・RHR・HRV)"])

if entry_type == "練習セッション記録 (ランTRIMP対応)":
    st.sidebar.subheader("練習セッションの追加")
    with st.sidebar.form("session_form"):
        s_date = st.date_input("練習日 (日付)", datetime.date.today())
        s_cat = st.selectbox("カテゴリ / 計算方式", list(CATEGORY_DEFAULTS.keys()))
        s_note = st.text_input("メニュー詳細・メモ", placeholder="例: 1000m×4本、ロードジョグ6km、乱取8本")

        s_load = 0.0
        calc_method = "Session-RPE"
        s_rpe_val = 5

        if s_cat == "ランニング（心拍計測: Banister法）":
            calc_method = "Banister-HR"
            st.markdown("---")
            st.caption("❤️ Banister予備心拍数TRIMP計算")
            c_r1, c_r2 = st.columns(2)
            with c_r1:
                r_dur = st.number_input("ランニング時間 (分)", min_value=1, max_value=300, value=40, step=5)
                r_hr_avg = st.number_input("平均心拍数 (bpm)", min_value=60, max_value=220, value=150)
            with c_r2:
                r_hr_max = st.number_input("最大心拍数 (25歳設定: 195bpm)", value=DEFAULT_HR_MAX)
                r_hr_rest = st.number_input("安静時心拍数 (記録平均)", value=float(avg_rhr_recorded))
            
            s_dur = r_dur
            s_load = calc_banister_trimp(r_dur, r_hr_avg, r_hr_rest, r_hr_max, gender="male")
            delta_hr_ratio = round((r_hr_avg - r_hr_rest) / (r_hr_max - r_hr_rest), 3) if r_hr_max > r_hr_rest else 0
            st.info(f"📊 予備心拍比: {delta_hr_ratio} ｜ 算出TRIMP: {s_load} AU")

        elif s_cat == "ランニング（ペース・環境係数法）":
            calc_method = "Pace-Coeff"
            st.markdown("---")
            p_dist = st.number_input("走行距離 (km)", min_value=0.5, max_value=50.0, value=5.0, step=0.5)
            p_pace = st.selectbox("ペース帯 (強度係数)", list(PACE_COEFFS.keys()))
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                p_surf = st.selectbox("路面", list(SURFACE_COEFFS.keys()), index=0)
            with c_p2:
                p_weat = st.selectbox("天候・気温", list(WEATHER_COEFFS.keys()), index=0)
            
            s_dur = st.number_input("所要時間 (分)", min_value=5, max_value=300, value=25, step=5)
            k_pace = PACE_COEFFS[p_pace]
            k_surf = SURFACE_COEFFS[p_surf]
            k_weat = WEATHER_COEFFS[p_weat]
            s_load = round(k_pace * p_dist * k_surf * k_weat * 10, 1)
            st.info(f"算出TRIMP: {s_load} AU")

        else:
            calc_method = "Session-RPE"
            s_dur = st.number_input("実施時間 (分)", min_value=5, max_value=300, value=60, step=5)
            def_rpe = CATEGORY_DEFAULTS[s_cat]["rpe"] if CATEGORY_DEFAULTS[s_cat]["rpe"] else 5
            s_rpe_val = st.slider("自覚的運動強度 RPE (CR-10)", min_value=1, max_value=10, value=def_rpe)
            s_load = round(s_dur * s_rpe_val, 1)
            st.info(f"算出負荷 (TRIMP): {s_load} AU")

        submitted = st.form_submit_button("セッションを登録")
        if submitted:
            new_sess = pd.DataFrame([{
                "date": s_date.strftime("%Y-%m-%d"),
                "category": s_cat,
                "duration": s_dur,
                "rpe": s_rpe_val,
                "load": s_load,
                "calc_method": calc_method,
                "note": s_note
            }])
            curr_sess = pd.read_csv(DATA_SESSIONS_FILE)
            updated = pd.concat([curr_sess, new_sess], ignore_index=True)
            updated.to_csv(DATA_SESSIONS_FILE, index=False)
            st.sidebar.success("セッションを登録しました！")
            st.rerun()

else:
    st.sidebar.subheader("朝のコンディション登録")
    with st.sidebar.form("daily_form"):
        d_date = st.date_input("測定日（今朝の日付）", datetime.date.today())
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            d_sleep_start = st.text_input("前夜の入眠 (HH:MM)", "21:30")
        with col_s2:
            d_sleep_end = st.text_input("今朝の起床 (HH:MM)", "05:30")
        
        d_calc_hours = calc_sleep_hours(d_sleep_start, d_sleep_end)
        st.caption(f"睡眠時間計算: 約 {d_calc_hours} 時間")

        d_weight = st.number_input("今朝の体重 (kg)", min_value=40.0, max_value=150.0, value=75.0, step=0.1)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            d_rhr = st.number_input("安静時心拍数 RHR (bpm)", min_value=35, max_value=120, value=52)
        with col_c2:
            d_hrv = st.number_input("心拍変動 HRV (rMSSD)", min_value=10, max_value=200, value=65)

        d_submitted = st.form_submit_button("コンディションを登録")
        if d_submitted:
            curr_daily = pd.read_csv(DATA_DAILY_FILE)
            d_str = d_date.strftime("%Y-%m-%d")
            if d_str in curr_daily["date"].values:
                curr_daily.loc[curr_daily["date"] == d_str, ["sleep_start", "sleep_end", "weight", "rhr", "hrv"]] = [
                    d_sleep_start, d_sleep_end, d_weight, d_rhr, d_hrv
                ]
            else:
                new_row = pd.DataFrame([{
                    "date": d_str,
                    "sleep_start": d_sleep_start,
                    "sleep_end": d_sleep_end,
                    "weight": d_weight,
                    "rhr": d_rhr,
                    "hrv": d_hrv
                }])
                curr_daily = pd.concat([curr_daily, new_row], ignore_index=True)
            curr_daily.to_csv(DATA_DAILY_FILE, index=False)
            st.sidebar.success(f"{d_str} を保存しました！")
            st.rerun()

with st.sidebar.expander("⚙️ Banisterモデル時定数・係数調整"):
    tau1 = st.slider("フィットネス時定数 (τ1)", 20, 60, 45)
    tau2 = st.slider("疲労時定数 (τ2)", 5, 25, 15)
    k1 = st.slider("フィットネス重み (k1)", 0.5, 2.0, 1.0, 0.1)
    k2 = st.slider("疲労重み (k2)", 1.0, 3.0, 2.0, 0.1)

sessions_df = pd.read_csv(DATA_SESSIONS_FILE)
daily_df = pd.read_csv(DATA_DAILY_FILE)
analyzed_df = compute_banister_model(daily_df, sessions_df, tau1=tau1, tau2=tau2, k1=k1, k2=k2)

latest = analyzed_df.iloc[-1]
st.subheader(f"📊 最新ステータス ({latest['date'].strftime('%Y-%m-%d')} 朝時点)")

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    perf = latest["performance"]
    status_icon = "🟢 絶好調" if perf > 0 else ("🟡 調整期/疲労蓄積" if perf > -50 else "🔴 疲労過多")
    kpi1.metric("Performance (調子)", f"{perf:.1f} AU", status_icon)

with kpi2:
    kpi2.metric("前夜睡眠 / 直近7日計", f"{latest['sleep_hours']}h", f"7日計: {latest['sleep_7d_total']}h")

with kpi3:
    hrv_delta = round(latest["hrv"] - latest["hrv_7d_avg"], 1)
    kpi3.metric("今朝のHRV", f"{latest['hrv']} ms", f"7日比 {hrv_delta:+} ms")

with kpi4:
    rhr_diff_mean = round(latest["rhr"] - avg_rhr_recorded, 1)
    kpi4.metric("今朝のRHR", f"{latest['rhr']} bpm", f"平均比 {rhr_diff_mean:+} bpm")

with kpi5:
    kpi5.metric("今朝の体重", f"{latest['weight']} kg")

st.markdown("---")

st.subheader("📈 フィットネス - 疲労 - パフォーマンス曲線")
fig1 = make_subplots(specs=[[{"secondary_y": True}]])
colors = ["#3182CE", "#E53E3E", "#38A169", "#D69E2E", "#805AD5", "#4A5568"]

if not sessions_df.empty:
    pivot_sess = sessions_df.copy()
    pivot_sess["date"] = pd.to_datetime(pivot_sess["date"])
    pivot_df = pivot_sess.pivot_table(index="date", columns="category", values="load", aggfunc="sum").fillna(0)
    pivot_df = pivot_df.reindex(analyzed_df["date"]).fillna(0)

    for i, cat in enumerate(pivot_df.columns):
        fig1.add_trace(
            go.Bar(
                x=pivot_df.index,
                y=pivot_df[cat],
                name=f"負荷: {cat}",
                marker_color=colors[i % len(colors)],
                opacity=0.65
            ),
            secondary_y=False
        )
else:
    fig1.add_trace(
        go.Bar(x=analyzed_df["date"], y=analyzed_df["day_trimp"], name="Day TRIMP", marker_color="rgba(160, 174, 192, 0.4)"),
        secondary_y=False
    )

fig1.add_trace(go.Scatter(x=analyzed_df["date"], y=analyzed_df["fitness"], name=f"Fitness (体力: τ1={tau1})", line=dict(color="#2B6CB0", width=2.5)), secondary_y=False)
fig1.add_trace(go.Scatter(x=analyzed_df["date"], y=analyzed_df["fatigue"], name=f"Fatigue (疲労: τ2={tau2})", line=dict(color="#C53030", width=2.5, dash="dash")), secondary_y=False)
fig1.add_trace(go.Scatter(x=analyzed_df["date"], y=analyzed_df["performance"], name=f"Performance (予測調子: k2={k2})", line=dict(color="#D69E2E", width=3.5)), secondary_y=True)

fig1.update_layout(
    height=480,
    barmode="stack",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    margin=dict(l=20, r=20, t=30, b=20)
)
fig1.update_yaxes(title_text="TRIMP / Fitness / Fatigue (AU)", secondary_y=False)
fig1.update_yaxes(title_text="Performance (AU)", secondary_y=True, zeroline=True, zerolinecolor="gray")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🛌 前日練習負荷と翌朝リカバリー（睡眠・HRV・RHR）の因果照合")
col_g1, col_g2 = st.columns(2)

with col_g1:
    fig_rec = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rec.add_trace(go.Scatter(x=analyzed_df["date"], y=analyzed_df["hrv"], mode="lines+markers", name="今朝のHRV (ms)", line=dict(color="#38A169")), secondary_y=False)
    fig_rec.add_trace(go.Scatter(x=analyzed_df["date"], y=analyzed_df["hrv_7d_avg"], mode="lines", name="HRV 7日平均", line=dict(color="#276749", dash="dot")), secondary_y=False)
    fig_rec.add_trace(go.Scatter(x=analyzed_df["date"], y=analyzed_df["rhr"], mode="lines+markers", name="今朝のRHR (bpm)", line=dict(color="#DD6B20")), secondary_y=True)
    fig_rec.add_trace(go.Scatter(x=analyzed_df["date"], y=[avg_rhr_recorded]*len(analyzed_df), mode="lines", name=f"全期間平均RHR ({avg_rhr_recorded})", line=dict(color="#A0AEC0", dash="dash")), secondary_y=True)
    fig_rec.update_layout(height=360, title="自律神経応答 (今朝のHRV・RHR)", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_rec.update_yaxes(title_text="HRV (ms)", secondary_y=False)
    fig_rec.update_yaxes(title_text="RHR (bpm)", secondary_y=True)
    st.plotly_chart(fig_rec, use_container_width=True)

with col_g2:
    fig_slp = make_subplots(specs=[[{"secondary_y": True}]])
    fig_slp.add_trace(go.Bar(x=analyzed_df["date"], y=analyzed_df["sleep_hours"], name="前夜睡眠時間 (h)", marker_color="#805AD5", opacity=0.7), secondary_y=False)
    fig_slp.add_trace(go.Scatter(x=analyzed_df["date"], y=analyzed_df["prev_day_trimp"], mode="lines+markers", name="前日の練習TRIMP (AU)", line=dict(color="#E53E3E", dash="dot")), secondary_y=True)
    fig_slp.update_layout(height=360, title="前日練習負荷 vs 前夜睡眠時間", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_slp.update_yaxes(title_text="睡眠時間 (h)", range=[0, 12], secondary_y=False)
    fig_slp.update_yaxes(title_text="前日TRIMP (AU)", secondary_y=True)
    st.plotly_chart(fig_slp, use_container_width=True)

st.subheader("📋 直近の練習セッション記録")
st.dataframe(sessions_df.sort_values("date", ascending=False).head(15), use_container_width=True, hide_index=True)
