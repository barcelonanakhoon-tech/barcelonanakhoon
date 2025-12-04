import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from utils import (
    load_stock_data,
    create_features,
    prepare_data,
    evaluate_model,
    plot_confusion_matrix,
    simulate_trading_strategy,
    calculate_buy_and_hold_return,
    plot_trading_results,
    print_trade_log
)
import numpy as np
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

def compare_trading_strategies(results_dict):
    """
    여러 트레이딩 전략의 수익률을 비교합니다. (main.py로 이동)
    """
    comparison_data = []

    for strategy_name, result in results_dict.items():
        comparison_data.append({
            'Strategy': strategy_name,
            'Initial Capital': f"${result['initial_capital']:,.0f}",
            'Final Value': f"${result['final_value']:,.0f}",
            'Total Return (%)': f"{result['total_return']:.2f}",
            'Num Trades': result.get('num_trades', 'N/A'),
            'Total Fees': f"${result.get('total_fees_paid', 0):,.0f}" if 'total_fees_paid' in result else 'N/A'
        })

    df = pd.DataFrame(comparison_data)

    print("\n" + "="*100)
    print("트레이딩 전략 수익률 비교")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")

    return df
def generate_readme(model_name, ticker, start_date, end_date, results_df, model_result, buy_and_hold_result, best_params, image_path='trading_results.png'):
    """
    분석 결과를 바탕으로 README.md 파일을 생성합니다.
    """
    print("\n--- README.md 파일 생성 시작 ---")

    model_return = model_result['total_return']
    bh_return = buy_and_hold_result['total_return']

    # 테스트 기간 추출
    test_start_date = model_result['dates'][0].strftime('%Y-%m-%d')
    test_end_date = model_result['dates'][-1].strftime('%Y-%m-%d')

    # 수익률 비교 분석 문장 생성
    if model_return > bh_return:
        analysis_text = (
            f"시뮬레이션 결과, **{model_name} 모델 기반 전략의 최종 수익률은 {model_return:.2f}%** 로, "
            f"같은 기간 동안의 **매수 후 보유(Buy & Hold) 전략 수익률({bh_return:.2f}%) 대비 {model_return - bh_return:.2f}%p 높은 성과**를 기록했습니다. "
            "이는 본 모델이 주가의 등락 방향을 유의미하게 예측하여, 시장 평균 수익률을 상회하는 초과 수익을 창출할 수 있는 가능성을 보여줍니다."
        )
    else:
        analysis_text = (
            f"시뮬레이션 결과, **{model_name} 모델 기반 전략의 최종 수익률은 {model_return:.2f}%** 로, "
            f"같은 기간 동안의 **매수 후 보유(Buy & Hold) 전략 수익률({bh_return:.2f}%) 대비 {abs(model_return - bh_return):.2f}%p 낮은 성과**를 기록했습니다. "
            "이는 현재 모델의 예측 정확도가 잦은 거래로 발생하는 비용(수수료 등)을 상쇄하고 시장 평균 수익률을 넘어서기에는 다소 부족하다는 것을 의미합니다."
        )

    # README 내용 구성
    readme_content = f"""
# 📈 {ticker} 주가 예측 및 투자 전략 백테스팅 보고서

## 1. 프로젝트 개요

본 프로젝트는 머신러닝 모델을 활용하여 특정 주식 종목({ticker})의 미래 주가 방향성을 예측하고, 이를 기반으로 한 자동 매매 전략의 유효성을 검증하는 것을 목표로 합니다.
단순히 주식을 매수하고 보유하는 전통적인 '매수 후 보유(Buy & Hold)' 전략과 모델의 예측을 따르는 '모델 기반 전략'의 성과를 비교 분석하여, 데이터 기반의 정량적 투자의 가능성을 탐색합니다.

## 2. 분석 환경

- **분석 종목**: 두산에너빌리티 ({ticker})
- **분석 기간**: {start_date} ~ {end_date}
- **데이터 출처**: Yahoo Finance (`yfinance` 라이브러리)

## 3. 모델링

### 3.1. 사용 모델

- **모델 종류**: XGBoost (eXtreme Gradient Boosting)
- **모델 설명**: 트리 기반 앙상블 기법으로, 분류 및 회귀 문제에서 높은 성능을 보이며 금융 시계열 예측에 널리 사용됩니다.
- **예측 대상**: 다음 거래일의 종가가 당일 종가보다 높을지('상승') 혹은 낮거나 같을지('하락')를 예측하는 이진 분류(Binary Classification) 문제를 해결합니다.

### 3.2. 특성 공학 (Feature Engineering)
모델의 예측 성능을 높이기 위해 원본 시계열 데이터(OHLCV)로부터 다음과 같은 기술적 분석 지표들을 파생 변수(특성)로 생성하여 사용했습니다.

- **가격 기반 지표**: 이동평균 (5, 10, 20, 50, 200일) 및 현재가와의 비율, 변동성 (5, 10, 20일 수익률 표준편차), RSI, MACD
- **거래량 기반 지표**: 거래량 이동평균 및 현재 거래량과의 비율
- **과거 수익률**: 과거 N일의 수익률 (Lag Features)

### 3.3. 모델 설계 및 하이퍼파라미터

#### 아키텍처 선택 이유
본 프로젝트에서는 **XGBoost(eXtreme Gradient Boosting)** 모델을 채택했습니다. XGBoost는 다음과 같은 장점 때문에 금융 시계열 예측 문제에 적합하다고 판단했습니다.
- **높은 예측 성능**: 여러 데이터 과학 경진대회에서 입증된 바와 같이, 정형 데이터에 대해 매우 뛰어난 성능을 보입니다.
- **과적합 방지**: 자체적으로 규제(Regularization) 기능을 포함하고 있어 과적합을 효과적으로 제어할 수 있습니다.
- **유연성 및 속도**: 병렬 처리를 지원하여 대용량 데이터에 대해서도 빠른 학습이 가능하며, 다양한 하이퍼파라미터 튜닝을 통해 모델을 세밀하게 조정할 수 있습니다.

#### 하이퍼파라미터 최적화
`GridSearchCV`를 사용하여 교차 검증을 통해 최적의 하이퍼파라미터 조합을 탐색했습니다. 최종적으로 선택된 하이퍼파라미터는 다음과 같습니다.
```json
{json.dumps(best_params, indent=4)}
```

## 4. 백테스팅 (Backtesting)

### 4.1. 테스트 기간

- **기간**: {test_start_date} ~ {test_end_date}

### 4.2. 비교 전략

- **모델 기반 전략 ({model_name})**: 200일 이동평균선을 기준으로 시장을 '강세장'과 '약세장'으로 구분합니다. 강세장에서는 모델의 상승 예측 시 매수하고, 약세장에서는 모델의 하락 예측 시 매도하여 추세를 추종하고 위험을 관리하는 전략을 사용합니다.
- **매수 후 보유 전략 (Buy & Hold)**: 테스트 기간 첫 거래일에 주식을 전량 매수하여 마지막 거래일까지 보유합니다.

### 4.3. 시뮬레이션 결과

{results_df.to_markdown(index=False)}

## 5. 결론 및 분석

{analysis_text}

### 포트폴리오 가치 변화

아래 그래프는 테스트 기간 동안 각 전략에 따른 포트폴리오 가치의 변화 추이를 보여줍니다.

![트레이딩 결과]({image_path})

## 6. 코드 품질

본 프로젝트는 다음과 같은 원칙을 준수하여 코드의 품질을 높이고자 노력했습니다.

- **가독성**: 의미 있는 변수명과 함수명을 사용하고, 코드를 기능별로 모듈화하여 전체 구조를 쉽게 파악할 수 있도록 구성했습니다. (`main.py`는 실행 흐름, `utils.py`는 보조 함수)
- **주석 및 설명**: 모든 함수에 상세한 Docstring을 작성하여 함수의 역할, 파라미터, 반환 값을 명확히 설명했습니다. 복잡한 로직에는 인라인 주석을 추가하여 코드의 이해를 돕습니다.
- **보고서 자동화**: 이 `README.md` 파일은 코드 실행 시 시뮬레이션 결과와 함께 자동으로 생성되어, 분석 과정의 재현성과 문서화 효율성을 높였습니다.

## 6. 향후 개선 방향

현재 모델의 성능을 더욱 향상시키기 위해 다음과 같은 접근을 고려할 수 있습니다.

- **하이퍼파라미터 최적화**: Grid Search, Bayesian Optimization 등을 통해 XGBoost 모델의 최적 하이퍼파라미터를 탐색하여 예측 정확도를 높일 수 있습니다.
- **다양한 모델 활용**: LSTM, GRU와 같은 딥러닝 시계열 모델을 도입하여 XGBoost 모델과 성능을 비교 분석할 수 있습니다.
- **거래 전략 고도화**: 단순 매수/매도 전략을 넘어, 손절매(Stop-loss), 변동성 돌파 등 다양한 거래 규칙을 적용하여 위험 관리를 강화하고 수익률을 개선할 수 있습니다.

"""

    # 파일 작성
    try:
        with open('readme.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print("--- README.md 파일 생성 완료 ---")
    except Exception as e:
        print(f"README.md 파일 생성 중 오류 발생: {e}")


def main():
    """
    메인 실행 함수
    """
    # --- 1. 설정 ---
    import json
    TICKER = '034020.KS'  # 두산에너빌리티 종목 코드
    START_DATE = '2020-01-01'
    MODEL_NAME = 'XGBoost_TrendFilter' # 전략 이름 변경

    # --- 2. 데이터 준비 ---
    print("--- 데이터 준비 시작 ---")
    data = load_stock_data(ticker=TICKER, start_date=START_DATE)
    featured_data = create_features(data)

    # 데이터 정제: inf 값을 0으로 대체하고, NaN 값을 다시 한번 제거
    featured_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    featured_data.dropna(inplace=True)

    # 데이터 분할
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(featured_data)
    print("--- 데이터 준비 완료 ---\n")

    # --- 3. XGBoost 모델 학습 ---
    print(f"--- {MODEL_NAME} 모델 학습 시작 ---")

    # GridSearchCV를 사용한 하이퍼파라미터 최적화
    print("--- 하이퍼파라미터 최적화 시작 (GridSearchCV) ---")
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.8, 0.9]
    }

    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42
    )

    # GridSearchCV 설정 (3-fold cross-validation)
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("--- 하이퍼파라미터 최적화 완료 ---")
    print(f"최적 하이퍼파라미터: {grid_search.best_params_}")

    # 최적의 모델로 설정
    xgb_model = grid_search.best_estimator_

    print(f"--- {MODEL_NAME} 모델 학습 완료 ---\n")

    # --- 4. 모델 평가 ---
    print("--- 모델 평가 시작 ---")

    # predict_proba를 사용하여 '상승(1)' 클래스에 대한 확률을 얻음
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    # 50% 확률을 기준으로 이진 예측 생성 (성능 평가용)
    y_pred_xgb = (y_pred_proba_xgb > 0.5).astype(int)

    xgb_results = evaluate_model(y_test, y_pred_xgb, model_name=MODEL_NAME)
    plot_confusion_matrix(y_test, y_pred_xgb, model_name=MODEL_NAME)
    print("--- 모델 평가 완료 ---\n")

    # --- 5. 트레이딩 시뮬레이션 ---
    print("--- 트레이딩 시뮬레이션 시작 ---")
    test_dates = X_test.index
    test_data = featured_data.loc[test_dates]
    test_actual_prices = test_data['Close']
    test_ma_200 = test_data['MA_200']

    # 모델 기반 전략
    trading_result_xgb = simulate_trading_strategy(
        predictions=y_pred_proba_xgb, # 확률값을 전달
        actual_prices=test_actual_prices,
        dates=test_dates, ma_200=test_ma_200.values,
        buy_threshold=0.5 # 강세장에서는 50% 확률만 넘어도 매수
    )
    # Buy and Hold 전략 (벤치마크)
    buy_and_hold_result = calculate_buy_and_hold_return(test_actual_prices)

    # 결과 비교
    IMAGE_SAVE_PATH = 'trading_results.png'
    all_strategy_results = {MODEL_NAME: trading_result_xgb, 'Buy & Hold': buy_and_hold_result}
    results_df = compare_trading_strategies(all_strategy_results)
    plot_trading_results(all_strategy_results, save_path=IMAGE_SAVE_PATH)
    print_trade_log(trading_result_xgb['trade_log'])

    # README.md 파일 생성
    end_date = datetime.now().strftime('%Y-%m-%d')
    generate_readme(MODEL_NAME, TICKER, START_DATE, end_date, results_df, trading_result_xgb, buy_and_hold_result, grid_search.best_params_, image_path=IMAGE_SAVE_PATH)

    print("--- 모든 과정 완료 ---")

if __name__ == "__main__":
    main()