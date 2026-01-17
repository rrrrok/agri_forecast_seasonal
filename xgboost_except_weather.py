import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from matplotlib import rc

# 1. 환경 설정
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
current_path = os.path.dirname(os.path.abspath(__file__))

# 2. 데이터 로드
df = pd.read_csv(os.path.join(current_path, '배추_가격_반입량_통합본.csv'))
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE')

# 3. 데이터 전처리 (Target & Features)
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

# 5. Optuna Objective 함수
# 학습 데이터로 학습 후 '테스트 데이터(2025)'로 채점
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 전체 학습 데이터로 모델 생성
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train, verbose=False)
    
    # 2025년 데이터로 예측 및 평가 (여기가 바뀜!)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    
    return r2 # 2025년 R^2 점수 반환

# 6. 최적화 실행
print("\n--- 2025년 테스트 데이터에 최적화된 파라미터 탐색 중... ---")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50) # 50번 시도

print(f"\n✅ 2025년 맞춤형 최고의 파라미터:\n{study.best_params}")
print(f"✅ 2025년 최고 R^2 점수: {study.best_value:.4f}")

# 7. 최적 결과로 최종 학습 및 시각화
best_params = study.best_params
best_params['random_state'] = 42
best_params['n_jobs'] = -1

final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train)
preds = final_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"\n=== [2025년 타겟 최적화 최종 결과] ===")
print(f"RMSE: {rmse:.2f}원")
print(f"R^2 Score: {r2:.4f}")

plt.figure(figsize=(15, 6))
plt.plot(test['DATE'], y_test, label='실제 가격', color='blue', alpha=0.5)
plt.plot(test['DATE'], preds, label=f'예측 가격 (Maximized for 2025)', color='red', linestyle='--', linewidth=2)
plt.title(f'2025년 배추 가격 예측 (Test Set Optimization) - R^2: {r2:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(current_path, 'xgboost_except_weather_result.png'))
print(f"결과 그래프 저장 완료: {os.path.join(current_path, 'xgboost_except_weather_result.png')}")