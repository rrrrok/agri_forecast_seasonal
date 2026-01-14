import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드
df_p = pd.read_csv('배추일별가격_전처리완료.csv', parse_dates=['DATE'])
df_v = pd.read_csv('배추일별반입량_전처리완료.csv', parse_dates=['DATE'])

# 2. 데이터 병합
df = pd.merge(df_p[['DATE', '평균가격']], df_v[['DATE', '총반입량']], on='DATE', how='inner').sort_values('DATE')

# 3. 음수~양수 시차 분석 (-7일부터 +7일까지)
lags = range(-7, 8)
correlations = []

for lag in lags:
    # shift(lag)에서 lag가 음수면 데이터를 위로 올립니다 (미래 데이터를 오늘로 가져옴)
    corr = df['평균가격'].corr(df['총반입량'].shift(lag))
    correlations.append(corr)

# 4. 결과 시각화
plt.figure(figsize=(12, 6))
plt.axvline(0, color='red', linestyle='--', alpha=0.5) # 오늘 기준선
plt.axhline(0, color='black', linewidth=1)
plt.plot(lags, correlations, marker='o', color='darkblue')

plt.title('Cross-Correlation: Price vs Volume (Lag -7 to +7)')
plt.xlabel('Lag (Days) [Negative: Future Vol, Positive: Past Vol]')
plt.ylabel('Correlation Coefficient')
plt.xticks(lags)
plt.grid(True, alpha=0.3)
plt.show()

# 5. 최적 시차 도출
min_corr_idx = np.argmin(correlations)
best_lag = lags[min_corr_idx]

print(f"최저 상관계수 시차: {best_lag}일 (상관계수: {correlations[min_corr_idx]:.4f})")