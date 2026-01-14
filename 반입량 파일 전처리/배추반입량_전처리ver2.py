import pandas as pd
import numpy as np

# 1. 파일 불러오기
file_path = '배추반입량_전처리ver1.csv'
df = pd.read_csv(file_path)

# 2. DATE 형식 변환 및 정렬
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').drop_duplicates('DATE')

# 3. 전체 날짜 범위 생성 (2018-01-01 ~ 2024-12-31)
full_range = pd.date_range(start='2018-01-01', end='2024-12-31', freq='D')
full_df = pd.DataFrame({'DATE': full_range})

# 4. 기존 데이터와 병합
df = pd.merge(full_df, df, on='DATE', how='left')

# 5. 품목명 및 0값 처리
df['품목명'] = df['품목명'].fillna('배추')
# 0으로 표시된 데이터는 보간을 위해 NaN으로 일시 변경
df[['총반입량', '전일', '전년']] = df[['총반입량', '전일', '전년']].replace(0, np.nan)

# 6. 보간 수행 (양방향 선형 보간)
df['총반입량'] = df['총반입량'].interpolate(method='linear', limit_direction='both')

# 7. 전일/전년 데이터 채우기 (기존 데이터 우선, 없으면 shift로 보충)
df['전일'] = df['전일'].fillna(df['총반입량'].shift(1))
df['전년'] = df['전년'].fillna(df['총반입량'].shift(365))

# 8. 남은 결측치 보간 (2018년 초기값 등)
df['전일'] = df['전일'].interpolate(method='linear', limit_direction='both')
df['전년'] = df['전년'].interpolate(method='linear', limit_direction='both')

# 9. 소수점 제거 (정수형 변환)
# 결측치를 모두 채웠으므로 정수형으로 변환 가능합니다.
cols_to_fix = ['총반입량', '전일', '전년']
df[cols_to_fix] = df[cols_to_fix].round(0).astype(int)

# 10. 결과 저장
output_path = '../배추일별반입량_전처리완료.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"처리 완료: {output_path} (소수점 제거됨)")