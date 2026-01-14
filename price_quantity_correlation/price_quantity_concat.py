import pandas as pd
import os

# 현재 스크립트 파일이 있는 경로를 기준으로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 데이터 로드
# os.path.join을 사용하여 스크립트와 같은 폴더에 있는 파일을 읽습니다.
df_price = pd.read_csv(os.path.join(BASE_DIR, '배추일별가격_전처리완료.csv'))
df_volume = pd.read_csv(os.path.join(BASE_DIR, '배추일별반입량_전처리완료.csv'))

# 2. DATE 컬럼 형식 변환 (병합을 위해 필수)
df_price['DATE'] = pd.to_datetime(df_price['DATE'])
df_volume['DATE'] = pd.to_datetime(df_volume['DATE'])

# 3. 컬럼명 변경 (중복 방지 및 식별 용이성)
# 가격 데이터 컬럼명 변경
df_price = df_price.rename(columns={
    '평균가격': '가격_평균',
    '전일': '가격_전일',
    '전년': '가격_전년'
})

# 반입량 데이터 컬럼명 변경
df_volume = df_volume.rename(columns={
    '총반입량': '반입량_총',
    '전일': '반입량_전일',
    '전년': '반입량_전년'
})

# 4. 데이터 병합 (DATE 기준, inner join으로 공통 날짜만 추출)
# 품목명은 양쪽에 다 있으므로 한쪽은 제거하거나 병합 시 처리합니다.
df_merged = pd.merge(
    df_price, 
    df_volume.drop(columns=['품목명']), # 품목명 중복 방지
    on='DATE', 
    how='inner'
)

# 5. 컬럼 순서 재정렬 (보기 편하게 정리)
cols = ['DATE', '품목명', '단위', '등급명', '가격_평균', '가격_전일', '가격_전년', '반입량_총', '반입량_전일', '반입량_전년']
df_merged = df_merged[cols]

# 6. 결과 저장
output_path = os.path.join(BASE_DIR, '배추_가격_반입량_통합본.csv')
df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"통합 완료! 파일이 저장되었습니다: {output_path}")
print(df_merged.head())