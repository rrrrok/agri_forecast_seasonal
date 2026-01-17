import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib import rc

# 1. 환경 설정 및 데이터 로드
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
current_path = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(current_path, '배추_가격_반입량_통합본.csv'))
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE')

# 2. 피처 엔지니어링 (기상 제외, 가격/반입량 집중)
df['Target_7days'] = df['가격_평균'].shift(-7)
df['Month'] = df['DATE'].dt.month
df['Weekday'] = df['DATE'].dt.weekday

# 과거 패턴 주입 (Lag & Rolling)
df['Price_Today'] = df['가격_평균']
df['Price_Lag1'] = df['가격_평균'].shift(1)
df['Price_MA7'] = df['가격_평균'].rolling(window=7).mean()
df['Volume_Today'] = df['반입량_총']
df['Volume_MA7'] = df['반입량_총'].rolling(window=7).mean()

df = df.dropna()

# 데이터 분할 (2018-2024 학습, 2025 테스트)
train = df[(df['DATE'] >= '2018-01-01') & (df['DATE'] <= '2024-12-31')]
test = df[(df['DATE'] >= '2025-01-01') & (df['DATE'] <= '2025-12-31')]

features = ['Month', 'Weekday', 'Price_Today', 'Price_Lag1', 'Price_MA7', 'Volume_Today', 'Volume_MA7']
X_train, y_train = train[features], train['Target_7days']
X_test, y_test = test[features], test['Target_7days']

# 3. 모델 라인업 정의
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=500, max_depth=8, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, random_state=42, verbose=-1)
}

# 4. 학습 및 평가 루프
results = []
preds_dict = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds_dict[name] = preds
    
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    results.append({"Model": name, "R2": r2, "RMSE": rmse, "MAE": mae})

# 5. 결과 비교 및 시각화
results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
print("\n=== [모델 대항전 결과] ===")
print(results_df.to_string(index=False))

plt.figure(figsize=(15, 7))
plt.plot(test['DATE'], y_test, label='실제 7일 뒤 가격', color='black', alpha=0.3, linewidth=3)
for name in results_df['Model'][:3]: # 상위 3개 모델 시각화
    plt.plot(test['DATE'], preds_dict[name], label=f'예측 ({name})')

plt.title('모델별 배추 가격 예측 성능 비교 (2025년)', fontsize=15)
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig(os.path.join(current_path, 'model_comparison_except_weather.png'))
print(f"\n[완료] 가장 우수한 모델: {results_df.iloc[0]['Model']}")