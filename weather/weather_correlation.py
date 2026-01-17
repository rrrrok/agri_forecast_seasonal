import os
import pandas as pd
import numpy as np

# 1. 데이터 로드
current_path = os.path.dirname(os.path.abspath(__file__))
df_price = pd.read_csv(os.path.join(current_path, '배추_가격_반입량_통합본.csv'), encoding='utf-8-sig')
df_weather = pd.read_csv(os.path.join(current_path, '지역별_기상_전처리본.csv'), encoding='utf-8-sig')

df_price['DATE'] = pd.to_datetime(df_price['DATE'])
df_weather['일시'] = pd.to_datetime(df_weather['일시'])

# 2. 계절별 대표 지역 설정
def get_rep_region(date):
    m = date.month
    if 3 <= m <= 5: return '춘천'
    elif 6 <= m <= 8: return '대관령'
    else: return '해남'

df_price['대표지역'] = df_price['DATE'].apply(get_rep_region)

# 3. 최적 시차 탐색 함수
def find_optimal_lag(price_df, weather_df, region_name, max_lag=90):
    # 해당 지역 데이터 필터링
    p_sub = price_df[price_df['대표지역'] == region_name].copy()
    w_sub = weather_df[weather_df['지점명'] == region_name].copy()
    
    results = []
    
    for lag in range(max_lag + 1):
        # 기상 데이터에 시차 적용 (과거의 날씨를 현재 날짜로 당김)
        w_temp = w_sub.copy()
        w_temp['일시'] = w_temp['일시'] + pd.Timedelta(days=lag)
        
        # 가격 데이터와 병합
        merged = pd.merge(p_sub[['DATE', '가격_평균']], w_temp[['일시', '평균기온(°C)']], 
                          left_on='DATE', right_on='일시', how='inner')
        
        if len(merged) > 30: # 최소 데이터 개수 보장
            corr = merged['가격_평균'].corr(merged['평균기온(°C)'])
            results.append({'lag': lag, 'corr': corr, 'abs_corr': abs(corr)})
            
    # 절대값이 가장 큰 순으로 정렬
    res_df = pd.DataFrame(results).sort_values(by='abs_corr', ascending=False)
    return res_df

# 4. 각 지역별 분석 실행
print("--- [지역별/시차별] 가격-기온 상관관계 분석 결과 ---")
for region in ['춘천', '대관령', '해남']:
    opt_lag_df = find_optimal_lag(df_price, df_weather, region)
    
    top = opt_lag_df.iloc[0]
    print(f"\n[{region}] 대표 지역:")
    print(f"  - 최적 시차: {int(top['lag'])}일 전")
    print(f"  - 최고 상관계수(r): {top['corr']:.4f}")
    
    # 상위 5개 시차 확인 (경향성 파악용)
    print("  - 상위 시차 후보:")
    print(opt_lag_df.head(5)[['lag', 'corr']].to_string(index=False))