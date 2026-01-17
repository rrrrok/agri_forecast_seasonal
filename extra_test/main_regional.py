import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import os

# 1. 환경 설정
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
current_path = os.path.dirname(os.path.abspath(__file__))

# 2. 데이터 로드 및 전처리
df_region = pd.read_csv(os.path.join(current_path, '지역_분석.csv'))
df_region['DATE'] = pd.to_datetime(df_region['DATE'])
df_region['Month'] = df_region['DATE'].dt.month

# 3. 주산지 분류
def classify_region(city):
    if '해남' in str(city): return '해남'
    elif '춘천' in str(city): return '춘천'
    elif any(x in str(city) for x in ['평창', '강릉', '정선', '태백']): return '대관령'
    else: return '기타'

df_region['Region_Group'] = df_region['산지-시군구'].apply(classify_region)

# 타겟 지역 필터링
df_target = df_region[df_region['Region_Group'].isin(['해남', '대관령', '춘천'])].copy()

# 4. 월별 집계
monthly_vol = df_target.groupby(['Month', 'Region_Group'])['총거래물량'].sum().reset_index()

# 5. 시각화 (선 그래프)
plt.figure(figsize=(12, 6))

sns.lineplot(
    data=monthly_vol, 
    x='Month', 
    y='총거래물량', 
    hue='Region_Group', 
    palette={'해남': '#FF6B6B', '대관령': '#4D96FF', '춘천': '#FFD93D'}, # 색상 지정
    marker='o', 
    linewidth=3,
    markersize=9
)

# 데코레이션
plt.title('월별 배추 메인 주산지 물량 흐름 (Line Chart)', fontsize=16, fontweight='bold')
plt.xlabel('월 (Month)', fontsize=12)
plt.ylabel('총 거래 물량 (톤)', fontsize=12)
plt.xticks(range(1, 13))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='주산지', loc='upper right')

# 교차 지점(Cross Point) 강조 텍스트
plt.axvline(x=6, color='gray', linestyle=':', alpha=0.5)
plt.text(6.1, monthly_vol['총거래물량'].max()*0.5, '🔄 6월: 주산지 교체\n(해남 → 대관령)', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(current_path, 'main_producing_area_line.png'))
print(f"그래프 저장 완료: {os.path.join(current_path, 'main_producing_area_line.png')}")