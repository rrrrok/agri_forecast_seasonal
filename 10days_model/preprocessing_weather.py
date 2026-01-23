import pandas as pd
import numpy as np
import os

def process_weather_data_mixed(file_path, output_name):
    # 1. 데이터 로드
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp949')

    # 2. 날짜 파싱 및 변수 생성
    df['date'] = pd.to_datetime(df['조회일자'], format='%Y%m%d')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # 3. '순' (상순, 중순, 하순) 구분
    def get_sun(day):
        if day <= 10:
            return '상순'
        elif day <= 20:
            return '중순'
        else:
            return '하순'
    
    df['sun_type'] = df['day'].apply(get_sun)
    sun_map = {'상순': 1, '중순': 2, '하순': 3}
    df['sun_idx'] = df['sun_type'].map(sun_map)
    
    # 4. [1차 집계] 일별 데이터 -> 순(10일) 데이터
    # 기온, 습도 -> 평균 (Mean)
    # 강수량, 일조, 일사 -> 합계 (Sum)
    
    # 컬럼 매핑 (원본 컬럼명 -> 집계 방식)
    agg_rules = {}
    
    # 데이터에 존재하는 컬럼만 룰에 추가
    if '평균 기온(°C)' in df.columns: agg_rules['평균 기온(°C)'] = 'mean'
    if '최고 기온(°C)' in df.columns: agg_rules['최고 기온(°C)'] = 'mean'
    if '최저 기온(°C)' in df.columns: agg_rules['최저 기온(°C)'] = 'mean'
    if '평균 습도(%)' in df.columns: agg_rules['평균 습도(%)'] = 'mean'
    
    if '평균 강수량(mm)' in df.columns: agg_rules['평균 강수량(mm)'] = 'sum'
    if '평균 일조시간(hr)' in df.columns: agg_rules['평균 일조시간(hr)'] = 'sum'
    if '평균 일사량(MJ/㎡)' in df.columns: agg_rules['평균 일사량(MJ/㎡)'] = 'sum'

    # 그룹화하여 1차 집계 수행
    df_sun = df.groupby(['year', 'month', 'sun_type', 'sun_idx']).agg(agg_rules).reset_index()
    
    # 5. 오름차순 정렬
    df_sun = df_sun.sort_values(by=['year', 'month', 'sun_idx'], ascending=[True, True, True]).reset_index(drop=True)
    
    # 6. 날짜 형식 컬럼 생성 (YYYYMM순)
    df_sun['일시'] = (
        df_sun['year'].astype(str) + 
        df_sun['month'].apply(lambda x: f"{x:02d}") + 
        df_sun['sun_type']
    )
    
    # 7. [2차 집계] 1순 ~ 30순 이동 집계 (Rolling Window)
    # 요청사항: 기온 -> 평균(Mean), 나머지(강수,일사,일조,습도) -> 누적(Sum)
    
    df_result = df_sun[['year', 'month', 'sun_type', '일시']].copy() # 기본 정보 복사
    
    # 컬럼별로 다르게 rolling 적용
    for col, rule in agg_rules.items():
        clean_col_name = col.split('(')[0].replace(' ', '')
        
        # 룰에 따른 접미사 결정 (명확성을 위해)
        # 기온은 '평균', 나머지는 '누적'이라고 명시하고 싶으시면 아래 로직 활용
        
        for i in range(1, 31):
            col_name = f'{clean_col_name}_{i}순'
            
            if '기온' in col:
                # 기온: 1순~30순 기간의 '평균'
                df_result[col_name] = df_sun[col].rolling(window=i, min_periods=1).mean()
            elif '습도' in col:
                 # 습도: 사용자가 '누적 습도'를 요청함 -> Sum
                df_result[col_name] = df_sun[col].rolling(window=i, min_periods=1).sum()
            else:
                # 강수량, 일사량, 일조시간: '누적' -> Sum
                df_result[col_name] = df_sun[col].rolling(window=i, min_periods=1).sum()

    # 8. 지역/지점 정보 추가
    try:
        region = df['지역(시군)'].iloc[0]
        branch = df['지점명'].iloc[0]
        df_result.insert(0, '지점명', branch)
        df_result.insert(0, '지역', region)
    except:
        pass

    # '일시' 컬럼 위치 조정
    cols = [c for c in df_result.columns if c not in ['일시', 'year', 'month', 'sun_type']]
    # 지역, 지점명, 일시 순서로 배치
    front_cols = []
    if '지역' in df_result.columns: front_cols.append('지역')
    if '지점명' in df_result.columns: front_cols.append('지점명')
    front_cols.append('일시')
    
    final_cols = front_cols + [c for c in cols if c not in front_cols]
    df_result = df_result[final_cols]

    # 9. 저장
    save_name = f"{output_name}.csv"
    df_result.to_csv(save_name, index=False, encoding='utf-8-sig')
    print(f"[{save_name}] 저장 완료! (기온: 평균, 그외: 누적)")

# --- 실행부 ---
files = [
    {'path': '대관령.csv', 'output': '대관령_순단위'},
    {'path': '해남.csv', 'output': '해남_순단위'}
]

for file in files:
    if os.path.exists(file['path']):
        process_weather_data_mixed(file['path'], file['output'])
    else:
        print(f"파일을 찾을 수 없습니다: {file['path']}")