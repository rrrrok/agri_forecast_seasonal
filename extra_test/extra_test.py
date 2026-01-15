import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 전처리
# 1. 데이터 로드 및 전처리
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '지역_분석.csv')
df = pd.read_csv(file_path, parse_dates=['DATE'])

# 2. 계절 분류 함수 정의
def get_season(month):
    if month in [3, 4, 5]: return '봄'
    elif month in [6, 7, 8]: return '여름'
    elif month in [9, 10, 11]: return '가을'
    else: return '겨울'

df['계절'] = df['DATE'].dt.month.apply(get_season)

# 3. 시각화 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 계절 순서 고정
seasons = ['봄', '여름', '가을', '겨울']

# 4. 계절별로 서브플롯 생성
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, season in enumerate(seasons):
    # 해당 계절 데이터 필터링
    season_df = df[df['계절'] == season]
    
    # 지역(광역시도)과 등급별 총 거래물량 합산 (피벗 테이블)
    pivot_df = season_df.pivot_table(index='산지-광역시도', 
                                     columns='등급', 
                                     values='총거래물량', 
                                     aggfunc='sum',
                                     fill_value=0)
    
    # 히트맵 그리기
    sns.heatmap(pivot_df, annot=True, fmt='.0f', cmap='YlGnBu', ax=axes[i])
    axes[i].set_title(f'[{season}] 지역별-등급별 거래물량 분포', fontsize=15)
    axes[i].set_xlabel('등급')
    axes[i].set_ylabel('생산 지역')

plt.tight_layout()
plt.show()