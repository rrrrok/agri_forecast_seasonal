import pandas as pd

import os

# 스크립트 파일의 절대 경로를 기준으로 데이터 파일 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '배추생산량_지역별.csv')

# 1. 데이터 로드 (첫 두 행이 연도와 세부항목이므로 멀티 인덱스로 읽기)
# 파일 인코딩은 보통 'cp949' 또는 'utf-8-sig'입니다.
df = pd.read_csv(file_path, encoding='cp949', header=[0, 1])

# 2. 분석을 위한 데이터 정제
# '시도별' 컬럼에서 '계' (전국 합계) 행은 제외하고 지역 데이터만 추출 
df_regions = df[df[('시도별', '시도별')] != '계'].copy()

# 숫자 데이터에 포함된 '-' 또는 결측치를 0으로 변환하고 숫자형으로 타입 변경
for col in df_regions.columns[1:]:
    df_regions[col] = pd.to_numeric(df_regions[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

# 3. 사용자 정의 계절 분류에 따른 생산량 합산 (2024년 기준) 
# 봄: 일반봄배추 + 노지봄배추
df_regions[('2024', '봄_합계')] = (
    df_regions[('2024', '일반봄배추:생산량 (톤)')] + 
    df_regions[('2024', '노지봄배추:생산량 (톤)')]
)

# 계절별 타겟 컬럼 매핑
seasonal_mapping = {
    '봄': ('2024', '봄_합계'),
    '여름': ('2024', '고랭지배추:생산량 (톤)'),
    '가을': ('2024', '노지가을배추:생산량 (톤)'),
    '겨울': ('2024', '노지겨울배추:생산량 (톤)')
}

# 4. 계절별 가중치(비중) 계산 및 출력
print("### 2024년 기준 계절별 지역 가중치 (비중) ###\n")

for season, col_name in seasonal_mapping.items():
    # 해당 계절의 전체 생산량
    total_prod = df_regions[col_name].sum()
    
    if total_prod > 0:
        # 지역별 비중 계산 (가중치)
        df_regions[(season, 'weight')] = df_regions[col_name] / total_prod
        
        # 비중이 높은 상위 지역만 필터링하여 출력
        weights = df_regions[[('시도별', '시도별'), (season, 'weight')]]
        weights.columns = ['지역', '가중치']
        top_weights = weights[weights['가중치'] > 0].sort_values(by='가중치', ascending=False)
        
        print(f"[{season}배추 주산지 비중]")
        print(top_weights.head(5).to_string(index=False))
        print("-" * 30)