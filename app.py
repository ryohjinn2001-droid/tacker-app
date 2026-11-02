"""
Training & Conditioning Tracker v3 (データリセット版)
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

# --- データを初期化（ダミーデータを空ファイルで上書き） ---
pd.DataFrame(columns=["date", "category", "duration", "rpe", "load", "calc_method", "note"]).to_csv(DATA_SESSIONS_FILE, index=False)
pd.DataFrame(columns=["date", "sleep_start", "sleep_end", "weight", "rhr", "hrv"]).to_csv(DATA_DAILY_FILE, index=False)

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
    if daily_df.empty:
        # 空のデータフレーム用ダミー行
        return pd.DataFrame([{
            "date": pd.to_datetime(datetime.date.today()),
            "sleep_hours": 0.0,
            "sleep_7d_total": 0.0,
            "hrv": 0,
            "hrv_7d_avg": 0.0,
            "rhr": 0,
            "rhr_mean": 0.0,
            "weight": 0.0,
            "day_trimp": 0.0,
            "prev_day_trimp": 0.0,
            "fitness": 0.0,
            "fatigue": 0.0,
            "performance": 0.0
        }])

    daily = daily_df.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    if not sessions_df.empty:
        sess = sessions_df.copy()
