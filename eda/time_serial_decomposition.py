import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 한글 깨짐 방지 설정 (Windows 기준: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드
df = pd.read_csv('배추_가격_반입량_통합본.csv', parse_dates=['DATE'])
df.set_index('DATE', inplace=True)

# 2. 시계열 분해
result = seasonal_decompose(df['가격_평균'], model='additive', period=365)

# 3. 가독성을 높인 커스텀 시각화
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# (1) Observed: 실제 가격 + 이동평균
axes[0].plot(df.index, df['가격_평균'], label='실제 가격', color='lightgray', alpha=0.7)
axes[0].plot(df.index, df['가격_평균'].rolling(window=30).mean(), label='30일 이동평균', color='red', lw=1.5)
axes[0].set_title('배추 가격 추이 (Observed)', fontsize=14)
axes[0].legend(loc='upper left')

# (2) Trend: 장기적인 상승/하락 흐름
axes[1].plot(result.trend, color='blue', lw=2)
axes[1].set_title('장기 추세 (Trend)', fontsize=14)

# (3) Seasonal: 연간 반복 패턴 (특정 구간 확대 추천)
axes[2].plot(result.seasonal, color='green')
axes[2].set_title('연간 계절성 (Seasonal)', fontsize=14)

# (4) Resid: 불규칙 변동 (노이즈)
axes[3].scatter(df.index, result.resid, color='black', s=2, alpha=0.5)
axes[3].set_title('불규칙 요인 (Residual)', fontsize=14)

plt.tight_layout()
plt.show()