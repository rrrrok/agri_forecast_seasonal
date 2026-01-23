import pandas as pd
import numpy as np
import xgboost as xgb
# import optuna  <-- Optuna 제거
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from matplotlib import rc

# 1. 환경 설정
# (참고: 실행 환경에 따라 폰트 설정이 다를 수 있습니다. Mac은 'AppleGothic', 리눅스는 설치된 한글 폰트 사용)
rc('font', family='Malgun Gothic') 
plt.rcParams['axes.unicode_minus'] = False
current_path = os.path.dirname(os.path.abspath(__file__))

# 2. 데이터 로드
# 파일이 같은 폴더에 있어야 합니다.
df = pd.read_csv(os.path.join(current_path, '배추_가격_반입량_통합본.csv'))
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE')

# 3. 데이터 전처리 (Target & Features)
# 7일 뒤 가격을 예측
df['Target_7days'] = df['가격_평균'].shift(-7)

df['Month'] = df['DATE'].dt.month
df['Weekday'] = df['DATE'].dt.weekday
df['Price_Today'] = df['가격_평균']
df['Price_1day_ago'] = df['가격_평균'].shift(1)
df['Price_7days_ago'] = df['가격_평균'].shift(7)
df['Volume_Today'] = df['반입량_총']
df['Volume_MA_7'] = df['반입량_총'].rolling(window=7).mean()
df['Price_MA_7'] = df['가격_평균'].rolling(window=7).mean()
df['Price_MA_30'] = df['가격_평균'].rolling(window=30).mean()

df = df.dropna()

# 4. 학습/테스트 분리
train = df[(df['DATE'] >= '2018-01-01') & (df['DATE'] <= '2024-12-31')]
test = df[(df['DATE'] >= '2025-01-01') & (df['DATE'] <= '2025-12-31')]

features = ['Month', 'Weekday', 'Price_Today', 'Price_1day_ago', 'Price_7days_ago', 
            'Volume_Today', 'Volume_MA_7', 'Price_MA_7', 'Price_MA_30']

X_train, y_train = train[features], train['Target_7days']
X_test, y_test = test[features], test['Target_7days']

print(f"학습 데이터: {X_train.shape} (2018-2024)")
print(f"테스트 데이터: {X_test.shape} (2025)")

# 5. 모델 파라미터 설정 (Optuna 대신 수동 설정)
# 일반적인 시계열/회귀 문제에서 성능이 좋고 안정적인 추천 파라미터입니다.
best_params = {
    'n_estimators': 1000,       # 트리의 개수 (너무 적으면 과소적합, 많으면 시간 오래 걸림)
    'learning_rate': 0.03,      # 학습률 (낮을수록 정교하지만 n_estimators를 늘려야 함)
    'max_depth': 6,             # 트리의 깊이 (보통 5~8 사이 사용)
    'min_child_weight': 1,
    'subsample': 0.8,           # 데이터 샘플링 비율 (과적합 방지)
    'colsample_bytree': 0.8,    # 컬럼 샘플링 비율 (과적합 방지)
    'gamma': 0.1,               # 리프 노드를 추가로 나눌지 결정하는 최소 손실 감소값
    'random_state': 42,
    'n_jobs': -1                # 가능한 모든 CPU 코어 사용
}

print(f"\n✅ 적용된 추천 파라미터:\n{best_params}")

# 6. 모델 학습 및 평가
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train)

# 예측
preds = final_model.predict(X_test)

# 평가 지표 계산
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"\n=== [2025년 타겟 최종 결과 (Manual Parameters)] ===")
print(f"RMSE: {rmse:.2f}원")
print(f"R^2 Score: {r2:.4f}")

# 7. 시각화
plt.figure(figsize=(15, 6))
plt.plot(test['DATE'], y_test, label='실제 가격', color='blue', alpha=0.5)
plt.plot(test['DATE'], preds, label=f'예측 가격 (Recommended Params)', color='red', linestyle='--', linewidth=2)
plt.title(f'2025년 배추 가격 예측 (Manual Tuning) - R^2: {r2:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

save_path = os.path.join(current_path, 'xgboost_manual_result.png')
plt.savefig(save_path)
print(f"결과 그래프 저장 완료: {save_path}")
plt.show()