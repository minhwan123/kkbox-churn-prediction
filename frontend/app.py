import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

API_URL = "http://127.0.0.1:8000"
AVG_CHURN_RATE = 0.09  # 학습 데이터 전체 평균 이탈률 (개별 고객과 비교용)

COLOR_STAY = "#0ca30c"
COLOR_CHURN = "#d03b3b"
RISK_COLOR = {"low": "🟢", "medium": "🟡", "high": "🔴"}
RISK_BADGE_COLOR = {"low": "#0ca30c", "medium": "#c98a00", "high": "#d03b3b"}
SEGMENT_COLOR = {"즉시 관리 필요": "#d03b3b", "모니터링 대상": "#eda100", "안정 고객": "#0ca30c"}

st.set_page_config(page_title="KKBox 고객 이탈 예측", page_icon="📉", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stVerticalBlockBorderWrapper"] {
        transition: box-shadow 0.2s ease, transform 0.15s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 6px 18px rgba(0,0,0,0.10);
        transform: translateY(-1px);
    }
    </style>
    <div style="background:linear-gradient(135deg,#2a78d6 0%,#4a3aa7 100%);
                border-radius:16px;padding:28px 32px;margin-bottom:18px;color:#ffffff;">
        <div style="font-size:28px;font-weight:800;">📉 KKBox 고객 이탈 예측</div>
        <div style="font-size:14px;opacity:0.9;margin-top:6px;">
            AI 기반 이탈 확률 예측 &amp; 리텐션 액션 자동 추천 서비스
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return np.log(p / (1 - p))


@st.cache_data(ttl=300)
def get_model_info():
    res = requests.get(f"{API_URL}/model-info", timeout=10)
    res.raise_for_status()
    return res.json()


try:
    model_info = get_model_info()
except requests.exceptions.RequestException:
    model_info = None

if model_info:
    st.caption(f"📈 이 모델의 검증 성능: ROC-AUC {model_info['roc_auc']:.3f} (테스트셋 기준)")
    with st.expander("🔍 모델은 전체적으로 어떤 요인을 중요하게 볼까? (전체 데이터 기준)"):
        gi_df = pd.DataFrame(model_info["global_importance"]).sort_values("importance")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(gi_df["feature"], gi_df["importance"], color="#2a78d6")
        ax.set_xlabel("전체 모델 기준 중요도 (정규화 비율)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, width=480)

FIELD_KEYS = [
    "bd_clean", "tx_count", "total_paid", "avg_plan_days", "cancel_count", "prior_plan_days",
    "log_days", "avg_secs_per_day", "avg_unq_per_day", "completion_ratio",
    "city", "gender", "registered_via", "prior_auto_renew", "is_new_customer",
]

PRESETS = {
    "고위험 고객 (자동갱신 꺼짐)": {
        "bd_clean": 45, "tx_count": 4, "total_paid": 1000, "avg_plan_days": 30.0,
        "cancel_count": 1, "prior_plan_days": 30.0,
        "log_days": 5, "avg_secs_per_day": 1500.0, "avg_unq_per_day": 8.0, "completion_ratio": 0.4,
        "city": "4", "gender": "female",
        "registered_via": "4", "prior_auto_renew": 0, "is_new_customer": 0,
    },
    "안정적인 고객 (자동갱신 켜짐)": {
        "bd_clean": 35, "tx_count": 24, "total_paid": 3600, "avg_plan_days": 30.0,
        "cancel_count": 0, "prior_plan_days": 30.0,
        "log_days": 60, "avg_secs_per_day": 6000.0, "avg_unq_per_day": 25.0, "completion_ratio": 0.75,
        "city": "1", "gender": "male",
        "registered_via": "7", "prior_auto_renew": 1, "is_new_customer": 0,
    },
    "신규 고객 (거래 이력 없음)": {
        "bd_clean": 24, "tx_count": 0, "total_paid": 0, "avg_plan_days": 0.0,
        "cancel_count": 0, "prior_plan_days": 0.0,
        "log_days": 0, "avg_secs_per_day": 0.0, "avg_unq_per_day": 0.0, "completion_ratio": 0.0,
        "city": "-1", "gender": "unknown",
        "registered_via": "9", "prior_auto_renew": 0, "is_new_customer": 1,
    },
}

# What-if 시뮬레이터에서 값을 바꿔볼 수 있는 요인과, 그 값을 스캔할 격자(grid)
WHATIF_OPTIONS = {
    "사전 결제 횟수 (tx_count)": ("tx_count", [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]),
    "사전 총 결제금액 (total_paid)": ("total_paid", [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]),
    "사전 해지신청 횟수 (cancel_count)": ("cancel_count", [0, 1, 2, 3, 4, 5]),
    "직전 요금제 기간 (prior_plan_days)": ("prior_plan_days", [0, 30, 60, 90, 120, 180, 240, 300, 360]),
    "나이 (bd_clean)": ("bd_clean", [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80]),
    "청취 활동 일수 (log_days)": ("log_days", [0, 10, 30, 60, 100, 150, 200, 300, 500, 700]),
    "일평균 청취 시간 (avg_secs_per_day)": ("avg_secs_per_day", [0, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]),
    "일평균 청취 곡 수 (avg_unq_per_day)": ("avg_unq_per_day", [0, 5, 10, 15, 20, 25, 30, 40, 50]),
    "완청 비율 (completion_ratio)": ("completion_ratio", [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    "자동갱신 여부 (prior_auto_renew)": ("prior_auto_renew", [0, 1]),
    "신규 고객 여부 (is_new_customer)": ("is_new_customer", [0, 1]),
}


def apply_preset():
    preset_name = st.session_state["preset_select"]
    if preset_name in PRESETS:
        for k, v in PRESETS[preset_name].items():
            st.session_state[k] = v


def build_payload_from_values(bd_clean, tx_count, total_paid, avg_plan_days, cancel_count,
                               prior_plan_days, log_days, avg_secs_per_day, avg_unq_per_day, completion_ratio,
                               city, gender, registered_via, prior_auto_renew, is_new_customer):
    # log_days가 0이면(관측 시점 이전 청취 기록 자체가 없으면) 나머지 청취 비율 피처는
    # 정의 자체가 불가능하므로, 화면에 어떤 값이 입력되어 있든 결측(None)으로 보낸다.
    has_logs = bool(log_days and log_days > 0)
    return {
        "bd_clean": bd_clean if bd_clean and bd_clean > 0 else None,
        "tx_count": tx_count,
        "total_paid": total_paid,
        "avg_plan_days": avg_plan_days,
        "cancel_count": cancel_count,
        "prior_plan_days": prior_plan_days,
        "log_days": log_days,
        "avg_secs_per_day": avg_secs_per_day if has_logs else None,
        "avg_unq_per_day": avg_unq_per_day if has_logs else None,
        "completion_ratio": completion_ratio if has_logs else None,
        "city": city,
        "gender": gender,
        "registered_via": registered_via,
        "prior_auto_renew": prior_auto_renew,
        "is_new_customer": is_new_customer,
    }


def render_prediction_result(result, base_payload, key_prefix):
    """/predict 응답 하나를 위험도 카드 + 추천 액션 + 고객 메시지 + SHAP 워터폴 + What-if 시뮬레이터로 렌더링한다.
    개별 고객 조회 탭과 배치 결과의 드릴다운 상세보기가 이 함수를 공유해서 사용한다."""
    prob = result["churn_probability"]
    risk = result["risk_level"]

    with st.container(border=True):
        m1, m2 = st.columns(2)
        m1.metric(
            "이탈 확률", f"{prob:.1%}",
            delta=f"{(prob - AVG_CHURN_RATE):+.1%} vs 전체 평균({AVG_CHURN_RATE:.0%})",
            delta_color="inverse",
        )
        badge_color = RISK_BADGE_COLOR[risk]
        m2.markdown(
            f"""
            <div style="text-align:right;margin-top:6px;">
                <span style="display:inline-block;padding:8px 22px;border-radius:999px;
                             background:{badge_color}1f;color:{badge_color};font-weight:800;
                             font-size:19px;border:2px solid {badge_color};">
                    {RISK_COLOR[risk]} {risk.upper()}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        gauge_fig, gauge_ax = plt.subplots(figsize=(4, 2.3), subplot_kw={"aspect": "equal"})
        zones = [(0.0, 0.2, COLOR_STAY), (0.2, 0.5, "#eda100"), (0.5, 1.0, COLOR_CHURN)]
        for start, end, color in zones:
            theta1 = 180 * (1 - end)
            theta2 = 180 * (1 - start)
            gauge_ax.add_patch(
                mpatches.Wedge((0.5, 0), 0.42, theta1, theta2, width=0.16, color=color, alpha=0.85)
            )
        needle_angle = np.radians(180 * (1 - min(prob, 1.0)))
        nx, ny = 0.5 + 0.32 * np.cos(needle_angle), 0.32 * np.sin(needle_angle)
        gauge_ax.plot([0.5, nx], [0, ny], color="#0b0b0b", linewidth=2.5, solid_capstyle="round", zorder=5)
        gauge_ax.scatter([0.5], [0], color="#0b0b0b", s=45, zorder=6)
        gauge_ax.text(0.5, -0.16, f"{prob:.1%}", ha="center", fontsize=20, fontweight="bold")
        gauge_ax.set_xlim(0, 1)
        gauge_ax.set_ylim(-0.22, 0.48)
        gauge_ax.axis("off")
        st.pyplot(gauge_fig, width=280)

    if model_info:
        with st.container(border=True):
            st.markdown("#### 📊 이 고객, 어디쯤 있을까?")
            col_dist, col_radar = st.columns(2)

            with col_dist:
                st.caption(f"전체 고객 중 상위 {100 - result['percentile']:.0f}% 위험군에 속해요.")
                hist = model_info["probability_histogram"]
                edges = hist["bin_edges"]
                counts = hist["counts"]
                centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
                bar_width = edges[1] - edges[0]
                fig, ax = plt.subplots(figsize=(4.2, 3))
                ax.bar(centers, counts, width=bar_width * 0.9, color="#c3c2b7")
                ax.axvline(prob, color=COLOR_CHURN, linewidth=2)
                ax.text(prob, max(counts) * 1.03, "현재 고객", color=COLOR_CHURN,
                        ha="center", fontsize=8, fontweight="bold")
                ax.set_xlabel("이탈 확률", fontsize=8)
                ax.set_ylabel("고객 수", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, width=380)

            with col_radar:
                st.caption("평균 고객(점선) 대비 이 고객(실선)의 주요 지표 비교")
                radar_keys = [
                    "bd_clean", "tx_count", "total_paid", "avg_plan_days", "cancel_count", "prior_plan_days",
                    "log_days", "completion_ratio",
                ]
                radar_labels = [
                    "나이", "결제 횟수", "총 결제금액", "평균 요금제 기간", "해지 횟수", "직전 요금제 기간",
                    "청취 활동일수", "완청 비율",
                ]
                radar_max = {
                    "bd_clean": 80, "tx_count": 40, "total_paid": 5000,
                    "avg_plan_days": 90, "cancel_count": 5, "prior_plan_days": 360,
                    "log_days": 400, "completion_ratio": 1.0,
                }

                current_vals = [(base_payload.get(k) or 0) for k in radar_keys]
                current_norm = [min(v / radar_max[k], 1.0) for v, k in zip(current_vals, radar_keys)]
                avg_vals = [model_info["reference_medians"][k] for k in radar_keys]
                avg_norm = [min(v / radar_max[k], 1.0) for v, k in zip(avg_vals, radar_keys)]

                angles = np.linspace(0, 2 * np.pi, len(radar_keys), endpoint=False).tolist()
                current_norm += current_norm[:1]
                avg_norm += avg_norm[:1]
                angles += angles[:1]

                fig, ax = plt.subplots(figsize=(4.2, 3.6), subplot_kw={"projection": "polar"})
                ax.plot(angles, avg_norm, color="#898781", linewidth=1.5, linestyle="--", label="평균 고객")
                ax.fill(angles, avg_norm, color="#898781", alpha=0.1)
                ax.plot(angles, current_norm, color="#2a78d6", linewidth=2, label="이 고객")
                ax.fill(angles, current_norm, color="#2a78d6", alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(radar_labels, fontsize=7)
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7, frameon=False)
                plt.tight_layout()
                st.pyplot(fig, width=380)

    with st.container(border=True):
        st.markdown("#### 🎯 추천 리텐션 액션")
        st.info(result["recommended_action"])

    with st.container(border=True):
        st.markdown("#### 📱 고객이 실제로 받게 될 화면")
        msg = result["customer_message"]
        if msg["channel"] is None:
            st.caption(msg["body"])
        else:
            coupon_html = ""
            if msg["coupon_code"]:
                coupon_html = f"""
                <div style="margin-top:10px;display:inline-block;border:1.5px dashed #2a78d6;
                            border-radius:8px;padding:6px 14px;font-weight:700;color:#2a78d6;
                            letter-spacing:1px;">
                    🎟️ {msg['coupon_code']}
                </div>
                """
            st.markdown(
                f"""
                <div style="max-width:380px;border:1px solid #e1e0d9;border-radius:16px;
                            padding:18px;background:#ffffff;box-shadow:0 2px 10px rgba(0,0,0,0.08);">
                    <div style="font-size:12px;color:#898781;margin-bottom:6px;">📲 {msg['channel']}</div>
                    <div style="font-size:16px;font-weight:700;margin-bottom:8px;">{msg['title']}</div>
                    <div style="font-size:14px;color:#333333;line-height:1.5;">{msg['body']}</div>
                    {coupon_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.markdown("#### 이 예측에 영향을 준 요인 (SHAP)")
        st.caption("모델의 기준값에서 출발해, 각 요인이 더해지며 최종 예측까지 어떻게 이동하는지 보여줍니다.")

        base_prob = result["base_probability"]
        steps = [("모델 기준값", None)]
        steps += [(c["feature"], c["contribution"]) for c in result["top_contributions"]]
        steps.append(("기타 요인", result["other_contribution"]))

        cum_logit = logit(base_prob)
        cum_probs = [base_prob]
        for _, shap_val in steps[1:]:
            cum_logit += shap_val
            cum_probs.append(sigmoid(cum_logit))

        labels = [s[0] for s in steps] + ["최종 예측"]
        cum_probs.append(prob)

        fig, ax = plt.subplots(figsize=(9.5, 3.6))
        n = len(labels)
        for i in range(n):
            if i == 0 or i == n - 1:
                ax.bar(i, cum_probs[i], color="#4a3aa7", width=0.6)
            else:
                prev_v, cur_v = cum_probs[i - 1], cum_probs[i]
                color = COLOR_CHURN if cur_v > prev_v else COLOR_STAY
                ax.bar(i, abs(cur_v - prev_v), bottom=min(prev_v, cur_v), color=color, width=0.6)
            ax.text(i, cum_probs[i] + 0.03, f"{cum_probs[i]:.0%}", ha="center", fontsize=7)
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
        ax.set_ylabel("이탈 확률", fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, width=760)

    with st.container(border=True):
        st.markdown("#### 🔬 What-if 시뮬레이터")
        st.caption("다른 조건은 그대로 두고, 하나의 요인만 바꿔보면 이탈 확률이 어떻게 달라지는지 확인해보세요.")

        whatif_label = st.selectbox(
            "바꿔볼 요인 선택", list(WHATIF_OPTIONS.keys()), key=f"{key_prefix}_whatif_feature"
        )
        whatif_key, grid_values = WHATIF_OPTIONS[whatif_label]

        if st.button("시뮬레이션 실행", key=f"{key_prefix}_whatif_run"):
            current_v = base_payload.get(whatif_key)
            values = sorted(set(grid_values + ([current_v] if current_v is not None else [])))
            probs = []
            try:
                with st.spinner("여러 시나리오 계산 중..."):
                    for v in values:
                        sim_payload = dict(base_payload)
                        sim_payload[whatif_key] = v
                        r = requests.post(f"{API_URL}/predict", json=sim_payload, timeout=10)
                        r.raise_for_status()
                        probs.append(r.json()["churn_probability"])

                fig, ax = plt.subplots(figsize=(6.5, 3))
                ax.plot(values, probs, marker="o", color="#2a78d6", linewidth=2)
                if current_v is not None and current_v in values:
                    idx = values.index(current_v)
                    ax.scatter([current_v], [probs[idx]], color=COLOR_CHURN, s=90, zorder=5, label="현재 값")
                    ax.legend(fontsize=7)
                ax.set_xlabel(whatif_label, fontsize=8)
                ax.set_ylabel("이탈 확률", fontsize=8)
                ax.set_ylim(0, 1)
                ax.tick_params(labelsize=7)
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, width=560)
            except requests.exceptions.RequestException as e:
                st.error(f"API 호출 실패: {e}")


tab_single, tab_batch = st.tabs(["개별 고객 조회", "CSV 일괄 예측"])

with tab_single:
    with st.container(border=True):
        st.subheader("고객 정보 입력")
        st.selectbox(
            "예시 고객으로 빠르게 채우기",
            ["직접 입력"] + list(PRESETS.keys()),
            key="preset_select",
            on_change=apply_preset,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("결제 이력")
            tx_count = st.number_input("사전 결제 횟수 (tx_count)", min_value=0, key="tx_count")
            total_paid = st.number_input("사전 총 결제금액 (total_paid)", min_value=0.0, key="total_paid")
            avg_plan_days = st.number_input("사전 평균 요금제 기간 (avg_plan_days)", min_value=0.0, key="avg_plan_days")
            cancel_count = st.number_input("사전 해지신청 횟수 (cancel_count)", min_value=0, key="cancel_count")
            prior_plan_days = st.number_input("직전 거래 요금제 기간 (prior_plan_days)", min_value=0.0, key="prior_plan_days")

        with col2:
            st.caption("청취 활동 (관측 시점 이전)")
            log_days = st.number_input("청취 활동 일수 (log_days, 없으면 0)", min_value=0, key="log_days")
            avg_secs_per_day = st.number_input("일평균 청취 시간(초) (avg_secs_per_day)", min_value=0.0, key="avg_secs_per_day")
            avg_unq_per_day = st.number_input("일평균 청취 곡 수 (avg_unq_per_day)", min_value=0.0, key="avg_unq_per_day")
            completion_ratio = st.number_input(
                "완청 비율 (completion_ratio, 0~1)", min_value=0.0, max_value=1.0, key="completion_ratio"
            )

        with col3:
            st.caption("인구통계 / 가입 정보")
            bd_clean = st.number_input("나이 (bd_clean, 모르면 0)", min_value=0, max_value=80, key="bd_clean")
            city = st.text_input("도시 코드 (city)", key="city")
            gender = st.selectbox("성별 (gender)", ["male", "female", "unknown"], key="gender")
            registered_via = st.text_input("가입 경로 코드 (registered_via)", key="registered_via")
            prior_auto_renew = st.selectbox("직전 거래 자동갱신 여부 (prior_auto_renew)", [1, 0], key="prior_auto_renew")
            is_new_customer = st.selectbox("신규 고객 여부 (is_new_customer)", [0, 1], key="is_new_customer")

        predict_clicked = st.button("이탈 확률 예측", type="primary")

    if predict_clicked:
        payload = build_payload_from_values(
            bd_clean, tx_count, total_paid, avg_plan_days, cancel_count, prior_plan_days,
            log_days, avg_secs_per_day, avg_unq_per_day, completion_ratio,
            city, gender, registered_via, prior_auto_renew, is_new_customer,
        )
        try:
            with st.spinner("예측 계산 중..."):
                res = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                res.raise_for_status()
                result = res.json()
            st.session_state["single_result"] = result
            st.session_state["single_payload"] = payload
        except requests.exceptions.RequestException as e:
            st.error(f"API 호출 실패: {e}")

    if "single_result" in st.session_state:
        render_prediction_result(st.session_state["single_result"], st.session_state["single_payload"], key_prefix="single")
    else:
        st.markdown(
            """
            <div style="text-align:center;padding:56px 24px;color:#898781;">
                <div style="font-size:42px;">🔍</div>
                <div style="font-size:15px;margin-top:10px;line-height:1.6;">
                    위에서 고객 정보를 입력하거나 예시 고객을 선택한 뒤<br>
                    <b>"이탈 확률 예측"</b> 버튼을 눌러보세요.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_batch:
    with st.container(border=True):
        st.subheader("CSV 업로드로 여러 고객 한 번에 예측")
        st.caption(
            "필수 컬럼: bd_clean, tx_count, total_paid, avg_plan_days, cancel_count, prior_plan_days, "
            "log_days, avg_secs_per_day, avg_unq_per_day, completion_ratio, "
            "city, gender, registered_via, prior_auto_renew, is_new_customer"
        )
        uploaded = st.file_uploader("CSV 파일 선택", type="csv")

    if uploaded is not None:
        try:
            with st.spinner("전체 고객 예측 계산 중..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                res = requests.post(f"{API_URL}/predict-batch", files=files, timeout=30)
                res.raise_for_status()
                result_df = pd.DataFrame(res.json())
            result_df = result_df.sort_values("churn_probability", ascending=False)
            st.session_state["batch_result_df"] = result_df
        except requests.exceptions.RequestException as e:
            st.error(f"API 호출 실패: {e}")

    if "batch_result_df" not in st.session_state:
        st.markdown(
            """
            <div style="text-align:center;padding:56px 24px;color:#898781;">
                <div style="font-size:42px;">📄</div>
                <div style="font-size:15px;margin-top:10px;line-height:1.6;">
                    위에서 고객 데이터가 담긴 CSV 파일을 업로드하면<br>
                    전체 현황과 세그먼트별 분석 결과를 볼 수 있어요.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "batch_result_df" in st.session_state:
        result_df = st.session_state["batch_result_df"]

        n_customers = len(result_df)
        avg_prob = result_df["churn_probability"].mean()
        n_high_risk = (result_df["risk_level"] == "high").sum()
        revenue_at_risk = result_df["expected_revenue_at_risk"].sum()

        with st.container(border=True):
            st.markdown("#### 📊 전체 현황")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("전체 고객 수", f"{n_customers:,}명")
            k2.metric("평균 이탈 확률", f"{avg_prob:.1%}")
            k3.metric("고위험 고객 수", f"{n_high_risk:,}명", f"{n_high_risk / n_customers:.1%}")
            k4.metric("예상 손실 매출", f"{revenue_at_risk:,.0f}", help="각 고객의 (이탈확률 × 사전 총 결제금액) 합산")

        with st.container(border=True):
            st.markdown("#### 🏷️ 관리 세그먼트")
            seg_counts = result_df["segment"].value_counts()
            seg_cols = st.columns(len(seg_counts))
            for col, (seg, cnt) in zip(seg_cols, seg_counts.items()):
                color = SEGMENT_COLOR.get(seg, "#2a78d6")
                with col:
                    st.markdown(
                        f"""
                        <div style="background:{color}18;border:1px solid {color}55;
                                    border-radius:12px;padding:14px;text-align:center;">
                            <div style="font-size:13px;color:{color};font-weight:700;">{seg}</div>
                            <div style="font-size:26px;font-weight:800;color:#0b0b0b;margin-top:4px;">
                                {cnt:,}명
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            selected_segments = st.multiselect(
                "세그먼트 필터", options=list(SEGMENT_COLOR.keys()), default=list(SEGMENT_COLOR.keys())
            )
            filtered_df = result_df[result_df["segment"].isin(selected_segments)]

        with st.container(border=True):
            st.markdown("#### 🗺️ 도시 × 가입경로별 평균 이탈확률")
            st.caption("셀 색이 진할수록 해당 그룹의 평균 이탈 확률이 높다는 뜻입니다.")
            pivot = result_df.pivot_table(index="city", columns="registered_via", values="churn_probability", aggfunc="mean")
            if pivot.size > 0:
                fig, ax = plt.subplots(figsize=(4, max(2, 0.28 * len(pivot.index))))
                im = ax.imshow(pivot.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(pivot.columns, fontsize=6)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index, fontsize=6)
                ax.set_xlabel("가입 경로 (registered_via)", fontsize=7)
                ax.set_ylabel("도시 코드 (city)", fontsize=7)
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        val = pivot.values[i, j]
                        if not np.isnan(val):
                            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=5.5,
                                     color="white" if val > 0.5 else "black")
                cbar = fig.colorbar(im, ax=ax, shrink=0.8)
                cbar.ax.set_title("평균\n이탈확률", fontsize=6, pad=6)
                cbar.ax.tick_params(labelsize=6)
                plt.tight_layout()
                st.pyplot(fig, width=480)
            else:
                st.caption("표시할 데이터가 부족합니다.")

        col_hist, col_top = st.columns([1.2, 1])
        with col_hist:
            with st.container(border=True):
                st.markdown("#### 이탈 확률 분포")
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.hist(result_df["churn_probability"], bins=20, color="#2a78d6", alpha=0.85)
                ax.axvline(0.2, color="#898781", linestyle="--", linewidth=1)
                ax.axvline(0.5, color="#898781", linestyle="--", linewidth=1)
                ax.set_xlabel("이탈 확률")
                ax.set_ylabel("고객 수")
                ax.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)

        with col_top:
            with st.container(border=True):
                st.markdown("#### 가장 위험한 고객 Top 10")
                top10 = result_df.head(10)[["churn_probability", "risk_level", "segment"]].reset_index(drop=True)
                top10.index += 1
                top10["churn_probability"] = top10["churn_probability"].map(lambda x: f"{x:.1%}")
                st.dataframe(top10, use_container_width=True)

        with st.container(border=True):
            st.markdown(f"#### 전체 결과 ({len(filtered_df):,}명 표시 중)")
            st.dataframe(filtered_df, use_container_width=True)
            st.download_button(
                "예측 결과 CSV 다운로드",
                filtered_df.to_csv(index=False).encode("utf-8-sig"),
                "churn_predictions.csv",
                "text/csv",
            )

        with st.container(border=True):
            st.markdown("#### 🔍 개별 고객 상세 보기")
            st.caption("목록에서 고객을 선택하면 SHAP 설명, 추천 액션, What-if 시뮬레이터를 바로 확인할 수 있어요.")

            drill_df = filtered_df.reset_index(drop=True)
            options = list(drill_df.index)

            def format_customer(i):
                r = drill_df.loc[i]
                return f"#{i + 1} · 이탈확률 {r['churn_probability']:.1%} · {r['segment']}"

            selected_i = st.selectbox("고객 선택", options, format_func=format_customer, key="batch_drill_select")

            if st.button("상세 보기", key="batch_drill_button"):
                row = drill_df.loc[selected_i]
                bd_val = float(row["bd_clean"]) if pd.notna(row["bd_clean"]) else None
                log_days_val = float(row["log_days"]) if pd.notna(row["log_days"]) else 0
                secs_val = float(row["avg_secs_per_day"]) if pd.notna(row["avg_secs_per_day"]) else 0.0
                unq_val = float(row["avg_unq_per_day"]) if pd.notna(row["avg_unq_per_day"]) else 0.0
                ratio_val = float(row["completion_ratio"]) if pd.notna(row["completion_ratio"]) else 0.0
                payload = build_payload_from_values(
                    bd_val, float(row["tx_count"]), float(row["total_paid"]), float(row["avg_plan_days"]),
                    float(row["cancel_count"]), float(row["prior_plan_days"]),
                    log_days_val, secs_val, unq_val, ratio_val,
                    str(row["city"]), str(row["gender"]), str(row["registered_via"]),
                    int(row["prior_auto_renew"]), int(row["is_new_customer"]),
                )
                try:
                    with st.spinner("상세 정보 계산 중..."):
                        res = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                        res.raise_for_status()
                        result = res.json()
                    st.session_state["batch_drill_result"] = result
                    st.session_state["batch_drill_payload"] = payload
                except requests.exceptions.RequestException as e:
                    st.error(f"API 호출 실패: {e}")

            if "batch_drill_result" in st.session_state:
                render_prediction_result(
                    st.session_state["batch_drill_result"], st.session_state["batch_drill_payload"], key_prefix="batch"
                )
