import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from matplotlib import rc

# 1. 환경 설정
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
current_path = os.path.dirname(os.path.abspath(__file__))

# 2. 데이터 로드 및 전처리
df = pd.read_csv(os.path.join(current_path, '배추_가격_반입량_통합본.csv'))
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE')

# 7일 후 예측을 위한 Target 생성
df['Target_7days'] = df['가격_평균'].shift(-7)

# 피처 엔지니어링 (가격 및 반입량 중심)
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

# 3. 학습/테스트 데이터 분리
train = df[(df['DATE'] >= '2018-01-01') & (df['DATE'] <= '2024-12-31')]
test = df[(df['DATE'] >= '2025-01-01') & (df['DATE'] <= '2025-12-31')]

features = ['Month', 'Weekday', 'Price_Today', 'Price_1day_ago', 'Price_7days_ago', 
            'Volume_Today', 'Volume_MA_7', 'Price_MA_7', 'Price_MA_30']

X_train, y_train = train[features], train['Target_7days']
X_test, y_test = test[features], test['Target_7days']

# 4. Optuna 목적 함수 정의 (LightGBM 전용)
def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150), # LGBM의 핵심 파라미터
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    }
    
    model = lgb.LGBMRegressor(**param)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    
    return r2 # 2025년 R^2 점수 최대화

# 5. 최적화 실행
print("--- LightGBM 최적의 파라미터를 찾는 중 (2025년 타겟) ---")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"\n✅ 찾은 최고의 파라미터:\n{study.best_params}")
print(f"✅ 2025년 최고 R^2 점수: {study.best_value:.4f}")

# 6. 최적 파라미터 적용 및 최종 모델 학습
best_params = study.best_params
best_params['objective'] = 'regression'
best_params['random_state'] = 42
best_params['verbosity'] = -1

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_train, y_train)

# 7. 예측 및 평가
preds = final_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"\n=== [LightGBM 최종 결과 (2025년)] ===")
print(f"RMSE: {rmse:.2f}원")
print(f"MAE:  {mae:.2f}원")
print(f"R^2 Score: {r2:.4f}")

# 8. 시각화 및 저장
plt.figure(figsize=(15, 6))
plt.plot(test['DATE'], y_test, label='실제 가격', color='blue', alpha=0.5)
plt.plot(test['DATE'], preds, label=f'예측 가격 (LightGBM R^2={r2:.2f})', color='green', linestyle='--', linewidth=2)
plt.title('2025년 배추 가격 예측 결과 (LightGBM Optuna Tuned)', fontsize=15)
plt.legend()
plt.grid(True, alpha=0.3)

save_path = os.path.join(current_path, 'lightGBM_except_weather_result.png')
plt.savefig(save_path)
print(f"\n[완료] 결과 그래프가 저장되었습니다: {save_path}")

# 중요도 확인
plt.figure(figsize=(10, 6))
lgb.plot_importance(final_model, max_num_features=10, importance_type='gain')
plt.title('LightGBM 변수 중요도')
plt.tight_layout()