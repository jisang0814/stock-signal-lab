from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_providers import detect_market, fetch_price_history

st.title("🧩 Portfolio Risk")
st.caption("포트폴리오 상관관계, 집중도, 리스크 기여도를 점검합니다.")

symbols_text = st.text_area(
    "종목 목록 (쉼표/줄바꿈 구분)",
    value="AAPL, MSFT, NVDA, 005930, 000660",
    height=90,
)
period = st.selectbox("조회 기간", ["3mo", "6mo", "1y", "2y"], index=2)
interval = st.selectbox("봉 주기", ["1d", "1wk"], index=0)
weights_text = st.text_input("가중치(선택, 동일 개수)", value="")
current_weights_text = st.text_input("현재 보유 비중(선택, 동일 개수)", value="")

raw_tokens = [t.strip().upper() for t in symbols_text.replace("\n", ",").split(",")]
symbols = list(dict.fromkeys([t for t in raw_tokens if t]))

if len(symbols) < 2:
    st.info("최소 2개 종목을 입력하세요.")
    st.stop()

@st.cache_data(ttl=600)
def _load_close(symbol: str, period: str, interval: str) -> pd.Series:
    market = detect_market(symbol)
    _, df = fetch_price_history(symbol, market, period=period, interval=interval)
    close_obj = df.get("close")
    if close_obj is None:
        raise ValueError("close 컬럼이 없습니다.")

    # 일부 데이터 소스 응답에서 close가 중복 컬럼(DataFrame)으로 들어오는 경우가 있어 첫 컬럼만 사용
    if isinstance(close_obj, pd.DataFrame):
        if close_obj.shape[1] == 0:
            raise ValueError("close 데이터가 비어 있습니다.")
        s = pd.to_numeric(close_obj.iloc[:, 0], errors="coerce")
    else:
        s = pd.to_numeric(close_obj, errors="coerce")

    s = s.dropna().copy()
    s.name = symbol
    return s

series = []
for s in symbols:
    try:
        series.append(_load_close(s, period, interval))
    except Exception:
        continue

if len(series) < 2:
    st.error("분석 가능한 종목이 부족합니다.")
    st.stop()

price_df = pd.concat(series, axis=1).dropna()
price_df = price_df.loc[:, ~price_df.columns.duplicated()].copy()
ret_df = price_df.pct_change().dropna()
if ret_df.empty:
    st.error("수익률 데이터가 부족합니다.")
    st.stop()

n = ret_df.shape[1]
if weights_text.strip():
    try:
        w = np.array([float(x.strip()) for x in weights_text.split(",")], dtype=float)
        if len(w) != n:
            raise ValueError("가중치 개수와 종목 수가 다릅니다.")
        if w.sum() <= 0:
            raise ValueError("가중치 합은 0보다 커야 합니다.")
        weights = w / w.sum()
    except Exception as exc:
        st.warning(f"가중치 파싱 실패({exc}), 동일가중 사용")
        weights = np.ones(n) / n
else:
    weights = np.ones(n) / n

target_vol = st.slider("목표 연환산 변동성(%)", 5.0, 60.0, 18.0, 0.5)
shrink_lambda = st.slider("공분산 Shrinkage λ", 0.0, 1.0, 0.25, 0.05)
max_weight_cap = st.slider("최대 종목 비중(%)", 5.0, 100.0, 35.0, 1.0) / 100.0
max_turnover = st.slider("최대 회전율(%)", 0.0, 100.0, 30.0, 1.0) / 100.0

corr = ret_df.corr()
cov = ret_df.cov() * 252
cov_raw = cov.values
diag_cov = np.diag(np.diag(cov_raw))
cov_shrunk = (1 - shrink_lambda) * cov_raw + shrink_lambda * diag_cov

port_var = float(weights @ cov_shrunk @ weights)
port_vol = float(np.sqrt(max(port_var, 0.0)) * 100)
ann_ret = float((ret_df.mean().values @ weights) * 252 * 100)
sharpe_like = ann_ret / port_vol if port_vol > 0 else 0.0

# Min-variance 기반 제안 가중치 (롱온리, 합=1)
cov_m = cov_shrunk
diag = np.clip(np.diag(cov_m), 1e-12, None)
inv_diag = 1.0 / diag
w_suggest = inv_diag / inv_diag.sum()


def _apply_weight_cap(w: np.ndarray, cap: float, max_iter: int = 20) -> np.ndarray:
    out = w.copy().astype(float)
    cap = float(np.clip(cap, 1e-6, 1.0))
    for _ in range(max_iter):
        over = out > cap
        if not np.any(over):
            break
        excess = float(np.sum(out[over] - cap))
        out[over] = cap
        under = ~over
        if not np.any(under):
            break
        room = np.maximum(cap - out[under], 0.0)
        room_sum = float(room.sum())
        if room_sum <= 0:
            break
        out[under] += excess * (room / room_sum)
    s = float(out.sum())
    if s > 0:
        out = out / s
    return out


w_suggest = _apply_weight_cap(w_suggest, max_weight_cap)

if current_weights_text.strip():
    try:
        cw = np.array([float(x.strip()) for x in current_weights_text.split(",")], dtype=float)
        if len(cw) != n:
            raise ValueError("현재 비중 개수와 종목 수가 다릅니다.")
        if cw.sum() <= 0:
            raise ValueError("현재 비중 합은 0보다 커야 합니다.")
        w_current = cw / cw.sum()
    except Exception as exc:
        st.warning(f"현재 비중 파싱 실패({exc}), 입력 비중을 현재 비중으로 사용")
        w_current = weights.copy()
else:
    w_current = weights.copy()


def _turnover(old_w: np.ndarray, new_w: np.ndarray) -> float:
    return float(0.5 * np.abs(new_w - old_w).sum())


def _apply_turnover_cap(old_w: np.ndarray, target_w: np.ndarray, cap: float) -> np.ndarray:
    t = _turnover(old_w, target_w)
    if t <= cap:
        return target_w
    if t <= 0:
        return old_w
    alpha = cap / t
    out = old_w + alpha * (target_w - old_w)
    s = float(out.sum())
    return out / s if s > 0 else out


w_suggest = _apply_turnover_cap(w_current, w_suggest, max_turnover)
actual_turnover = _turnover(w_current, w_suggest)

suggest_var = float(w_suggest @ cov_m @ w_suggest)
suggest_vol = float(np.sqrt(max(suggest_var, 0.0)) * 100)
scale = (target_vol / suggest_vol) if suggest_vol > 0 else 0.0
gross_exposure = max(scale, 0.0)

asset_vol = np.sqrt(np.clip(np.diag(cov_m), 1e-12, None)) * 100
risk_budget = (1 / np.clip(asset_vol, 1e-9, None))
risk_budget = risk_budget / risk_budget.sum()

hhi = float(np.sum(weights ** 2))
effective_n = float(1 / hhi) if hhi > 0 else 0.0

marginal = cov_shrunk @ weights
contrib_var = weights * marginal
contrib_pct = contrib_var / contrib_var.sum() * 100 if contrib_var.sum() != 0 else np.zeros_like(contrib_var)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("연환산 기대수익", f"{ann_ret:+.2f}%")
with k2:
    st.metric("연환산 변동성", f"{port_vol:.2f}%")
with k3:
    st.metric("Sharpe-like", f"{sharpe_like:.2f}")
with k4:
    st.metric("유효 종목수", f"{effective_n:.2f}")

st.metric("집중도(HHI)", f"{hhi:.4f}")

s1, s2, s3 = st.columns(3)
with s1:
    st.metric("제안 포트폴리오 변동성", f"{suggest_vol:.2f}%")
with s2:
    st.metric("목표 변동성", f"{target_vol:.2f}%")
with s3:
    st.metric("권장 총 익스포저", f"{gross_exposure:.2f}x")
st.caption(
    f"Shrinkage λ={shrink_lambda:.2f}, 비중 캡={max_weight_cap*100:.1f}%, "
    f"회전율 제한={max_turnover*100:.1f}% (실제 {actual_turnover*100:.1f}%)"
)

st.subheader("상관관계 히트맵")
fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1)
fig.update_layout(height=520)
st.plotly_chart(fig, use_container_width=True)

st.subheader("리스크 기여도")
risk_df = pd.DataFrame(
    {
        "symbol": ret_df.columns,
        "weight": weights,
        "risk_contribution_pct": contrib_pct,
    }
).sort_values("risk_contribution_pct", ascending=False)
st.dataframe(risk_df, use_container_width=True, hide_index=True)

st.subheader("제안 가중치 (Min-Variance 근사)")
suggest_df = pd.DataFrame(
    {
        "symbol": ret_df.columns,
        "current_weight": weights,
        "as_is_weight": w_current,
        "suggest_weight": w_suggest,
        "risk_budget_weight": risk_budget,
        "asset_vol_pct": asset_vol,
    }
).sort_values("suggest_weight", ascending=False)
st.dataframe(suggest_df, use_container_width=True, hide_index=True)

st.download_button(
    "제안 가중치 CSV 다운로드",
    data=suggest_df.to_csv(index=False).encode("utf-8"),
    file_name="suggested_weights.csv",
    mime="text/csv",
    use_container_width=True,
)

st.markdown("#### 복사용 제안 가중치 (symbol,weight)")
copy_text = "\n".join(
    f"{row.symbol},{row.suggest_weight:.6f}"
    for row in suggest_df.itertuples(index=False)
)
st.code(copy_text, language="text")
