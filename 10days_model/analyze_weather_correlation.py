import pandas as pd
import numpy as np
import os

def analyze_correlations():
    print("="*70)
    print("최적 기상 상관관계 분석 (2018년 제외)")
    print("="*70)

    # 1. 데이터 로드
    df_price = pd.read_csv('가격데이터_순별.csv', encoding='utf-8-sig')
    df_daegwallyeong = pd.read_csv('대관령_순단위.csv', encoding='utf-8-sig')
    df_haenam = pd.read_csv('해남_순단위.csv', encoding='utf-8-sig')

    # 2. 날짜 파싱 함수
    def parse_date_kor(date_str):
        # 202401상순 -> date 객체 (편의상 상순=05, 중순=15, 하순=25일로 매핑하여 시계열 처리)
        date_str = str(date_str)
        year = int(date_str[:4])
        month = int(date_str[4:6])
        period = date_str[6:]
        day = {'상순': 5, '중순': 15, '하순': 25}[period]
        return pd.Timestamp(year=year, month=month, day=day)

    df_price['date_obj'] = df_price['DATE'].apply(parse_date_kor)
    df_daegwallyeong['date_obj'] = df_daegwallyeong['일시'].apply(parse_date_kor)
    df_haenam['date_obj'] = df_haenam['일시'].apply(parse_date_kor)

    # 3. 2018년 데이터 제외
    df_price = df_price[df_price['date_obj'].dt.year > 2018].copy()
    
    # 기상 데이터도 2018년 이후 데이터만 사용하여 병합 (필요시 lag 생성을 위해 일부 이전 데이터 필요할 수 있으나, 
    # 여기서는 병합 기준이 가격 데이터이므로 가격 데이터가 2019년부터 있으면 자연스럽게 필터링됨)
    print(f"분석 기간: {df_price['date_obj'].min().date()} ~ {df_price['date_obj'].max().date()}")

    # 4. 주산지 기준 기상 데이터 병합
    # 12~5월: 해남, 6~11월: 대관령
    
    # 빈 데이터프레임 생성 (가격 데이터 기준)
    df_merged = df_price[['DATE', 'date_obj', '평균가격']].copy()
    
    # 월 정보 추출
    df_merged['month'] = df_merged['date_obj'].dt.month
    
    # 기상 데이터 매핑을 위한 딕셔너리 생성 (date_obj -> row)
    weather_map_daegwallyeong = df_daegwallyeong.set_index('date_obj').to_dict('index')
    weather_map_haenam = df_haenam.set_index('date_obj').to_dict('index')
    
    # 기상 변수 컬럼 가져오기 (누적 변수들)
    weather_cols = [c for c in df_haenam.columns if '순' in c and c not in ['일시', '지역', '지점명', 'sun_type']]
    
    # 병합 수행
    weather_data_list = []
    
    for _, row in df_merged.iterrows():
        curr_date = row['date_obj']
        month = row['month']
        
        # 주산지 결정
        if month in [12, 1, 2, 3, 4, 5]:
            source = '해남'
            data_map = weather_map_haenam
        else:
            source = '대관령'
            data_map = weather_map_daegwallyeong
            
        # 해당 날짜의 기상 데이터 찾기
        if curr_date in data_map:
            weather_row = data_map[curr_date]
            weather_row['source_region'] = source
            weather_data_list.append(weather_row)
        else:
            # 매칭되는 기상 데이터가 없는 경우 (결측)
            empty_row = {col: np.nan for col in weather_cols}
            empty_row['source_region'] = source
            weather_data_list.append(empty_row)
            
    df_weather_combined = pd.DataFrame(weather_data_list)
    
    # 인덱스 리셋 후 병합
    df_final = pd.concat([df_merged.reset_index(drop=True), df_weather_combined.reset_index(drop=True)], axis=1)
    
    # 5. 상관관계 분석 (Lag 적용)
    # 현재 가격 vs 과거 기상 (Lag 0 ~ 12순)
    
    results = []
    
    # 분석할 기본 변수 유형 (예: 평균기온, 강수량 등)
    # 컬럼명 형식이 '{변수명}_{n}순_누적' 또는 '{변수명}_{n}순' 이므로 이를 파싱
    base_vars = set()
    for col in weather_cols:
        # '_숫자순' 앞부분을 기본 변수명으로 간주
        parts = col.split('_')
        if len(parts) >= 2:
            base_var = parts[0]
            base_vars.add(base_var)
            
    print(f"\n분석 변수: {', '.join(base_vars)}")
    print("상관관계 계산 중...", end='', flush=True)

    for base_var in base_vars:
        # 각 변수에 대해 1~30순 누적 컬럼 확인
        for n in range(1, 31):
            col_name = f"{base_var}_{n}순" # preprocess_weather.py에서 생성한 이름 규칙 확인 필요
            # preprocess.py에서는 f'{clean_col_name}_{i}순' 으로 저장함 (뒤에 _누적 없음, 코드 확인 결과)
            
            # 실제 컬럼명이 존재하는지 확인 (혹시 모를 불일치 대비)
            # preprocess.py 로그: "기온: 평균, 그외: 누적" -> 이름은 그냥 _n순 으로 통일됨
            if col_name not in df_final.columns:
                continue
                
            # Lag 0 ~ 12 적용
            for lag in range(13):
                # 기상 데이터를 lag 만큼 shift (과거 데이터를 현재 행으로 가져옴)
                # lag=1 이면, 1순 전의 기상 데이터를 현재 가격과 매칭
                shifted_weather = df_final[col_name].shift(lag)
                
                # 상관관계 계산
                corr = df_final['평균가격'].corr(shifted_weather)
                
                results.append({
                    'Variable': base_var,
                    'Cumulative_Period': n,
                    'Lag': lag,
                    'Correlation': corr,
                    'Abs_Correlation': abs(corr)
                })
        print(".", end='', flush=True)
                
    print(" 완료!")
    
    # 6. 결과 정리 및 출력
    results_df = pd.DataFrame(results)
    results_df = results_df.dropna().sort_values('Abs_Correlation', ascending=False)
    
    print("\n" + "="*70)
    print("Top 10 기상 변수 조합 (높은 상관관계 순)")
    print("="*70)
    print(f"{'변수명':<15} | {'누적기간(순)':<10} | {'시차(Lag)':<10} | {'상관계수'}")
    print("-" * 60)
    
    for i in range(min(10, len(results_df))):
        row = results_df.iloc[i]
        print(f"{row['Variable']:<15} | {int(row['Cumulative_Period']):<10}순 | {int(row['Lag']):<10}순 | {row['Correlation']:.4f}")
        
    # 변수별 최적 조합 확인
    print("\n" + "="*70)
    print("각 변수별 최적 조합")
    print("="*70)
    print(f"{'변수명':<15} | {'누적기간':<10} | {'시차':<10} | {'상관계수'}")
    print("-" * 60)
    
    for base_var in base_vars:
        var_best = results_df[results_df['Variable'] == base_var].iloc[0]
        print(f"{base_var:<15} | {int(var_best['Cumulative_Period']):<10}순 | {int(var_best['Lag']):<10}순 | {var_best['Correlation']:.4f}")

    # 결과 저장
    results_df.to_csv('optimal_weather_correlation_results.csv', index=False, encoding='utf-8-sig')
    print("\n상세 분석 결과 저장 완료: optimal_weather_correlation_results.csv")

if __name__ == "__main__":
    analyze_correlations()
