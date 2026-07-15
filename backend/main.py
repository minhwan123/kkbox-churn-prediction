import io
import math
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

NUMERIC = [
    "bd_clean", "tx_count", "total_paid", "avg_plan_days", "cancel_count", "prior_plan_days",
    "log_days", "avg_secs_per_day", "avg_unq_per_day", "completion_ratio",
]
CATEGORICAL = ["city", "gender", "registered_via"]
BINARY = ["prior_auto_renew", "is_new_customer"]
FEATURE_COLUMNS = NUMERIC + CATEGORICAL + BINARY

# 02_preprocessing.ipynb에서 city/registered_via는 결측으로 인해 float 컬럼이 된 상태에서
# astype(str)을 거쳐 "4.0", "-1.0" 같은 문자열로 학습되었다. API로 들어오는 "4" 같은 입력을
# 그대로 두면 인코더가 모르는 범주로 처리해(handle_unknown="ignore") 해당 피처가 통째로 무시되므로,
# 항상 학습 시와 동일한 float-문자열 형식으로 정규화해야 한다.
NUMERIC_CODE_COLUMNS = ["city", "registered_via"]


def normalize_code(value) -> str:
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return str(value)


FEATURE_LABELS = {
    "bd_clean": "나이",
    "tx_count": "사전 결제 횟수",
    "total_paid": "사전 총 결제금액",
    "avg_plan_days": "사전 평균 요금제 기간",
    "cancel_count": "사전 해지신청 횟수",
    "prior_plan_days": "직전 요금제 기간",
    "prior_auto_renew": "자동갱신 여부",
    "is_new_customer": "신규 고객 여부",
    "city": "도시 코드",
    "gender": "성별",
    "registered_via": "가입 경로",
    "log_days": "청취 활동 일수",
    "avg_secs_per_day": "일평균 청취 시간",
    "avg_unq_per_day": "일평균 청취 곡 수",
    "completion_ratio": "완청 비율",
}

ACTION_BY_FEATURE = {
    "자동갱신 여부": "자동갱신 유도 프로모션 발송 (자동갱신 전환 할인 쿠폰)",
    "사전 해지신청 횟수": "CS 우선 상담 배정 (해지 이력 고객 전담 케어)",
    "신규 고객 여부": "온보딩 캠페인 대상 등록 (첫 결제 후 이용 가이드 발송)",
    "사전 결제 횟수": "장기 구독 혜택 안내 (구독 기간 연장 시 리워드)",
    "사전 총 결제금액": "VIP 리텐션 케어 (고액 결제 고객 전담 매니저 배정)",
    "직전 요금제 기간": "요금제 변경 제안 (더 긴 약정 요금제 할인 안내)",
    "사전 평균 요금제 기간": "요금제 변경 제안 (더 긴 약정 요금제 할인 안내)",
    "나이": "연령대 맞춤 콘텐츠 추천 발송",
    "성별": "맞춤 콘텐츠 추천 발송",
    "도시 코드": "지역 기반 오프라인 프로모션 안내",
    "가입 경로": "가입 채널별 맞춤 리텐션 메시지 발송",
    "청취 활동 일수": "휴면 고객 재활성화 캠페인 (맞춤 플레이리스트 추천 발송)",
    "일평균 청취 시간": "휴면 고객 재활성화 캠페인 (맞춤 플레이리스트 추천 발송)",
    "일평균 청취 곡 수": "추천 알고리즘 기반 신곡/플레이리스트 안내",
    "완청 비율": "콘텐츠 추천 개선 안내 (취향 재설정 유도)",
}

# "값이 낮아서 위험한 경우"와 "값이 높아서 위험한 경우"가 정반대 의미를 갖는 피처는
# SHAP 기여 방향만으로 구분할 수 없어, 실제 값이 이 임계치 미만이면 low-변형 액션/메시지를 사용한다.
LABEL_TO_RAW = {v: k for k, v in FEATURE_LABELS.items()}
LOW_VALUE_THRESHOLDS = {"total_paid": 500, "tx_count": 5}

ACTION_BY_FEATURE_LOW = {
    "사전 총 결제금액": "첫 구독 특별 혜택 안내 (초기 결제 유도 프로모션)",
    "사전 결제 횟수": "온보딩 캠페인 대상 등록 (이용 초반 이탈 방지)",
}


def is_low_value_driver(base_feature: str, row: pd.DataFrame) -> bool:
    raw_col = LABEL_TO_RAW.get(base_feature)
    threshold = LOW_VALUE_THRESHOLDS.get(raw_col)
    if threshold is None:
        return False
    return float(row[raw_col].iloc[0]) < threshold


def recommend_action(risk_level: str, top_contributions: list, row: pd.DataFrame) -> str:
    """위험도와 SHAP 최상위 위험 증가 요인을 바탕으로 구체적인 리텐션 액션을 추천한다."""
    if risk_level == "low":
        return "특별 조치 불필요 (안정 고객, 정기 모니터링만 진행)"

    driver = next((c for c in top_contributions if c.contribution > 0), None)
    if driver is None:
        return "모니터링 리스트 등록 (뚜렷한 위험 요인 없음)"

    base_feature = driver.feature.split(" = ")[0]
    if is_low_value_driver(base_feature, row):
        action = ACTION_BY_FEATURE_LOW.get(base_feature, ACTION_BY_FEATURE.get(base_feature, "CS 상담 배정"))
    else:
        action = ACTION_BY_FEATURE.get(base_feature, "CS 상담 배정")
    prefix = "[우선순위 높음]" if risk_level == "high" else "[모니터링]"
    return f"{prefix} {action}"


# 위험 요인별로 고객에게 실제 노출될 메시지(푸시 알림/이메일) 콘텐츠
CUSTOMER_MESSAGE_BY_FEATURE = {
    "자동갱신 여부": {
        "channel": "푸시 알림",
        "title": "🎵 끊김 없이 계속 들으세요!",
        "body": "지금 자동갱신을 설정하시면 다음 달 요금 50% 할인을 드려요.",
        "coupon_code": "AUTORENEW50",
    },
    "사전 해지신청 횟수": {
        "channel": "앱 인앱 메시지",
        "title": "😟 불편한 점이 있으셨나요?",
        "body": "고객님을 위한 전담 상담사가 배정되었어요. 무엇이든 편하게 말씀해주세요.",
        "coupon_code": None,
    },
    "신규 고객 여부": {
        "channel": "이메일",
        "title": "👋 가입을 환영합니다!",
        "body": "KKBox를 200% 즐기는 방법을 알려드릴게요. 첫 달 이용권 안내서를 확인해보세요.",
        "coupon_code": "WELCOME30",
    },
    "사전 결제 횟수": {
        "channel": "푸시 알림",
        "title": "🎁 오래 함께해주셔서 감사해요",
        "body": "그동안의 이용에 감사드리며, 특별 리워드 포인트를 지급해드렸어요.",
        "coupon_code": "LOYALTY20",
    },
    "사전 총 결제금액": {
        "channel": "이메일",
        "title": "👑 VIP 고객님을 위한 안내",
        "body": "고객님을 위한 전담 매니저가 배정되었습니다. 궁금한 점을 언제든 문의해주세요.",
        "coupon_code": "VIPCARE",
    },
    "직전 요금제 기간": {
        "channel": "푸시 알림",
        "title": "💰 더 저렴하게 이용하는 방법",
        "body": "장기 요금제로 바꾸면 매달 더 저렴해요. 지금 바로 확인해보세요.",
        "coupon_code": "LONGPLAN15",
    },
    "사전 평균 요금제 기간": {
        "channel": "푸시 알림",
        "title": "💰 더 저렴하게 이용하는 방법",
        "body": "장기 요금제로 바꾸면 매달 더 저렴해요. 지금 바로 확인해보세요.",
        "coupon_code": "LONGPLAN15",
    },
    "나이": {
        "channel": "푸시 알림",
        "title": "🎧 취향저격 플레이리스트 도착!",
        "body": "고객님을 위해 엄선한 추천 플레이리스트가 준비됐어요.",
        "coupon_code": None,
    },
    "성별": {
        "channel": "푸시 알림",
        "title": "🎧 취향저격 플레이리스트 도착!",
        "body": "고객님을 위해 엄선한 추천 플레이리스트가 준비됐어요.",
        "coupon_code": None,
    },
    "도시 코드": {
        "channel": "이메일",
        "title": "📍 우리 동네 특별 이벤트",
        "body": "거주 지역 대상 오프라인 이벤트 소식을 전해드려요.",
        "coupon_code": "LOCALEVT",
    },
    "가입 경로": {
        "channel": "푸시 알림",
        "title": "🎶 다시 만나 반가워요!",
        "body": "고객님을 위한 특별 혜택을 준비했어요.",
        "coupon_code": "WELCOMEBACK",
    },
    "청취 활동 일수": {
        "channel": "푸시 알림",
        "title": "🎧 요즘 뜸하셨네요!",
        "body": "고객님 취향에 맞는 신곡 플레이리스트를 준비했어요. 한 번 들어보세요.",
        "coupon_code": None,
    },
    "일평균 청취 시간": {
        "channel": "푸시 알림",
        "title": "🎧 요즘 뜸하셨네요!",
        "body": "고객님 취향에 맞는 신곡 플레이리스트를 준비했어요. 한 번 들어보세요.",
        "coupon_code": None,
    },
    "일평균 청취 곡 수": {
        "channel": "푸시 알림",
        "title": "🎶 새로운 음악을 만나보세요",
        "body": "고객님이 좋아할 만한 신곡과 플레이리스트를 추천해드려요.",
        "coupon_code": None,
    },
    "완청 비율": {
        "channel": "앱 인앱 메시지",
        "title": "🎯 취향에 안 맞으셨나요?",
        "body": "추천 알고리즘을 다시 맞춰드릴게요. 좋아하는 아티스트/장르를 선택해보세요.",
        "coupon_code": None,
    },
}

CUSTOMER_MESSAGE_BY_FEATURE_LOW = {
    "사전 총 결제금액": {
        "channel": "이메일",
        "title": "🎁 첫 구독을 응원해요!",
        "body": "지금 결제하시면 첫 달 특별 할인을 받으실 수 있어요.",
        "coupon_code": "FIRSTPAY30",
    },
    "사전 결제 횟수": {
        "channel": "이메일",
        "title": "👋 가입을 환영합니다!",
        "body": "KKBox를 200% 즐기는 방법을 알려드릴게요. 첫 달 이용권 안내서를 확인해보세요.",
        "coupon_code": "WELCOME30",
    },
}

DEFAULT_CUSTOMER_MESSAGE = {
    "channel": "앱 인앱 메시지",
    "title": "😊 안녕하세요, 고객님",
    "body": "고객님께 도움이 될 만한 소식을 준비했어요.",
    "coupon_code": None,
}

NO_ACTION_MESSAGE = {
    "channel": None,
    "title": "발송 없음",
    "body": "안정 고객으로 분류되어 별도 메시지를 발송하지 않습니다.",
    "coupon_code": None,
}


def build_customer_message(risk_level: str, top_contributions: list, row: pd.DataFrame) -> dict:
    """위험도와 SHAP 최상위 위험 요인을 바탕으로, 고객이 실제로 받게 될 메시지 콘텐츠를 생성한다."""
    if risk_level == "low":
        return dict(NO_ACTION_MESSAGE)

    driver = next((c for c in top_contributions if c.contribution > 0), None)
    if driver is None:
        return dict(DEFAULT_CUSTOMER_MESSAGE)

    base_feature = driver.feature.split(" = ")[0]
    if is_low_value_driver(base_feature, row):
        message = CUSTOMER_MESSAGE_BY_FEATURE_LOW.get(base_feature) or CUSTOMER_MESSAGE_BY_FEATURE.get(base_feature)
    else:
        message = CUSTOMER_MESSAGE_BY_FEATURE.get(base_feature)
    return dict(message or DEFAULT_CUSTOMER_MESSAGE)


MODEL_PATH = "../models/kkbox_churn_model.pkl"

app = FastAPI(title="KKBox Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = joblib.load(MODEL_PATH)
preprocess = pipe.named_steps["prep"]
classifier = pipe.named_steps["clf"]
explainer = shap.TreeExplainer(classifier)

# 레이더 차트에서 "평균 고객"과 비교하기 위한 대표값. 파이프라인의 SimpleImputer(strategy="median")가
# 이미 학습 데이터의 중앙값을 갖고 있으므로 별도 집계 없이 재사용한다.
NUMERIC_MEDIANS = dict(
    zip(NUMERIC, preprocess.named_transformers_["num"].named_steps["impute"].statistics_)
)

# 분포 위치/백분위 표시를 위해, 학습 데이터에서 5,000명을 샘플링해 이탈확률을 미리 계산해둔다.
_reference_sample = pd.read_parquet("../data/train_processed.parquet", columns=FEATURE_COLUMNS)
_reference_sample = _reference_sample.sample(min(5000, len(_reference_sample)), random_state=42)
REFERENCE_PROBS = np.sort(pipe.predict_proba(_reference_sample)[:, 1])
_hist_counts, _hist_edges = np.histogram(REFERENCE_PROBS, bins=20, range=(0, 1))
PROBABILITY_HISTOGRAM = {"counts": _hist_counts.tolist(), "bin_edges": _hist_edges.tolist()}

# TreeExplainer의 shap_values는 log-odds(margin) 공간에서 계산되므로,
# base_value(전체 평균 고객 기준 예측치)도 같은 공간에서 꺼내 sigmoid로 확률 변환해둔다.
BASE_VALUE_LOGIT = float(np.ravel(explainer.expected_value)[0])
BASE_VALUE_PROB = 1 / (1 + math.exp(-BASE_VALUE_LOGIT))


class CustomerFeatures(BaseModel):
    bd_clean: Optional[float] = Field(None, description="정제된 나이 (10~80, 결측 가능)")
    tx_count: float = Field(..., description="사전 결제 횟수")
    total_paid: float = Field(..., description="사전 총 결제금액")
    avg_plan_days: float = Field(..., description="사전 평균 요금제 기간(일)")
    cancel_count: float = Field(..., description="사전 해지신청 횟수")
    prior_plan_days: float = Field(..., description="직전 거래 요금제 기간(일)")
    log_days: float = Field(..., description="관측 시점 이전 청취 활동 일수")
    avg_secs_per_day: Optional[float] = Field(None, description="일평균 청취 시간(초), 청취 기록 없으면 결측 가능")
    avg_unq_per_day: Optional[float] = Field(None, description="일평균 청취 고유 곡 수, 청취 기록 없으면 결측 가능")
    completion_ratio: Optional[float] = Field(None, description="완청 비율(0~1), 청취 기록 없으면 결측 가능")
    city: str = Field(..., description="도시 코드 (예: '1', 결측은 '-1')")
    gender: str = Field(..., description="'male' / 'female' / 'unknown'")
    registered_via: str = Field(..., description="가입 경로 코드 (예: '7', 결측은 '-1')")
    prior_auto_renew: int = Field(..., ge=0, le=1, description="직전 거래 자동갱신 여부 (0/1)")
    is_new_customer: int = Field(..., ge=0, le=1, description="사전 거래 이력이 없는 신규 고객 여부 (0/1)")


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class CustomerMessage(BaseModel):
    channel: Optional[str]
    title: str
    body: str
    coupon_code: Optional[str]


class GlobalFeatureImportance(BaseModel):
    feature: str
    importance: float


class ProbabilityHistogram(BaseModel):
    counts: list[int]
    bin_edges: list[float]


class ModelInfo(BaseModel):
    roc_auc: float
    global_importance: list[GlobalFeatureImportance]
    reference_medians: dict[str, float]
    probability_histogram: ProbabilityHistogram


class PredictionResult(BaseModel):
    churn_probability: float
    risk_level: Literal["low", "medium", "high"]
    base_probability: float
    percentile: float
    top_contributions: list[FeatureContribution]
    other_contribution: float
    recommended_action: str
    customer_message: CustomerMessage


def to_risk_level(prob: float) -> str:
    if prob < 0.2:
        return "low"
    if prob < 0.5:
        return "medium"
    return "high"


def compute_global_importance(top_n: int = 8) -> list[dict]:
    """XGBoost의 전체 학습 기준 feature_importances_를 원본 피처 단위로 합산해,
    '모델이 전체적으로 무엇을 중요하게 보는지'를 정규화된 비율로 반환한다."""
    names = preprocess.get_feature_names_out()
    importances = classifier.feature_importances_

    agg: dict[str, float] = {}
    for name, imp in zip(names, importances):
        prefix, raw_col = name.split("__", 1)
        col = raw_col.rsplit("_", 1)[0] if prefix == "cat" else raw_col
        label = FEATURE_LABELS.get(col, col)
        agg[label] = agg.get(label, 0.0) + float(imp)

    total = sum(agg.values()) or 1.0
    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"feature": f, "importance": v / total} for f, v in ranked]


GLOBAL_IMPORTANCE = [GlobalFeatureImportance(**item) for item in compute_global_importance()]
MODEL_ROC_AUC = 0.8511  # 03_modeling.ipynb 테스트셋 기준 XGBoost 모델 성능 (청취 활동 피처 추가 후)


def explain_row(row: pd.DataFrame, top_n: int = 6) -> tuple[list[FeatureContribution], float]:
    """단일 행에 대해 SHAP 기여도를 계산하고, 원핫 인코딩된 '선택되지 않은' 범주는 제외한 뒤
    영향력이 큰 순서로 top_n개는 사람이 읽을 수 있는 이름과 함께, 나머지는 합산 기여도(other_contribution)로 반환한다."""
    transformed = preprocess.transform(row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    # ColumnTransformer가 만드는 희소 행렬은 XGBoost가 0을 '결측'으로 취급하는 채로 학습/예측되었으므로,
    # 밀집 배열로 바꾼 뒤에도 0을 NaN으로 되돌려야 실제 churn_probability와 일치하는 SHAP 값을 얻는다.
    # (그대로 두면 0을 '진짜 0'으로 취급해 전혀 다른 마진값을 계산하는 불일치가 생긴다.)
    shap_input = np.where(transformed == 0, np.nan, transformed)
    shap_values = explainer.shap_values(shap_input)[0]
    names = preprocess.get_feature_names_out()

    contributions = []
    for name, value, shap_val in zip(names, transformed[0], shap_values):
        prefix, raw_col = name.split("__", 1)
        if prefix == "cat":
            # 원핫 인코딩 컬럼 중 실제로 선택된(값=1) 범주만 사용
            if value != 1:
                continue
            col = raw_col.rsplit("_", 1)[0]
            display_value = row[col].iloc[0]
            if isinstance(display_value, str) and display_value.endswith(".0"):
                display_value = display_value[:-2]
            label = f"{FEATURE_LABELS.get(col, col)} = {display_value}"
        else:
            label = FEATURE_LABELS.get(raw_col, raw_col)
        contributions.append((label, float(shap_val)))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top = contributions[:top_n]
    # SHAP의 efficiency 성질(모든 기여도 합 + base_value = 모델 출력)을 이용해,
    # top_n에 포함되지 않은 나머지 요인들의 총 기여도를 정확히 역산한다.
    other_contribution = float(shap_values.sum()) - sum(c for _, c in top)
    return [FeatureContribution(feature=f, contribution=c) for f, c in top], other_contribution


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict(customer: CustomerFeatures):
    data = customer.model_dump()
    for col in NUMERIC_CODE_COLUMNS:
        data[col] = normalize_code(data[col])
    row = pd.DataFrame([data], columns=FEATURE_COLUMNS)
    prob = float(model_predict_proba(row))
    risk = to_risk_level(prob)
    contributions, other_contribution = explain_row(row)
    percentile = float(np.searchsorted(REFERENCE_PROBS, prob) / len(REFERENCE_PROBS) * 100)
    return PredictionResult(
        churn_probability=prob,
        risk_level=risk,
        base_probability=BASE_VALUE_PROB,
        percentile=percentile,
        top_contributions=contributions,
        other_contribution=other_contribution,
        recommended_action=recommend_action(risk, contributions, row),
        customer_message=CustomerMessage(**build_customer_message(risk, contributions, row)),
    )


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    return ModelInfo(
        roc_auc=MODEL_ROC_AUC,
        global_importance=GLOBAL_IMPORTANCE,
        reference_medians=NUMERIC_MEDIANS,
        probability_histogram=ProbabilityHistogram(**PROBABILITY_HISTOGRAM),
    )


def model_predict_proba(row: pd.DataFrame) -> float:
    return pipe.predict_proba(row)[:, 1][0]


@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드 가능합니다.")

    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"누락된 컬럼: {missing}")

    for col in CATEGORICAL:
        df[col] = df[col].astype(str)

    # 예측에는 학습 시와 동일한 정규화된 코드("4.0" 등)를 쓰되, 결과 테이블/다운로드에는
    # 사용자가 업로드한 원래 표기("4")를 그대로 보여주기 위해 별도 사본에만 정규화를 적용한다.
    encode_df = df.copy()
    for col in NUMERIC_CODE_COLUMNS:
        encode_df[col] = encode_df[col].apply(normalize_code)

    proba = pipe.predict_proba(encode_df[FEATURE_COLUMNS])[:, 1]
    result = df.copy()
    result["churn_probability"] = proba
    result["risk_level"] = [to_risk_level(p) for p in proba]

    SEGMENT_BY_RISK = {"high": "즉시 관리 필요", "medium": "모니터링 대상", "low": "안정 고객"}
    result["segment"] = result["risk_level"].map(SEGMENT_BY_RISK)
    result["expected_revenue_at_risk"] = result["churn_probability"] * result["total_paid"]

    return result.to_dict(orient="records")
