import os
import pandas as pd
import numpy as np

# 1. 경로 설정 (방법 3: 실행 파일 기준 자동 감지)
current_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_path, '지역별_기상.csv')
output_file = os.path.join(current_path, '지역별_기상_전처리본.csv')

# 2. 데이터 로드 (UTF-8 확정)
try:
    # 엑셀에서 저장한 UTF-8의 경우 'utf-8-sig'가 가장 안전합니다.
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    print(f"[성공] {input_file} 로드 완료")
except FileNotFoundError:
    print(f"[오류] 파일을 찾을 수 없습니다. 경로를 확인하세요: {input_file}")
    exit()

# 3. 결측치 보완 (Data Imputation)

# (1) 강수량: NaN은 비가 안 온 것이므로 0으로 채움
if '일강수량(mm)' in df.columns:
    df['일강수량(mm)'] = df['일강수량(mm)'].fillna(0)

# (2) 기온, 습도, 일조시간: 장비 오류 누락값은 지점별 선형 보간(Linear Interpolation)
# 지점명으로 그룹화하여 '대관령'의 빈 값을 '해남' 값으로 채우지 않도록 방지
cols_to_interpolate = ['평균기온(°C)', '최저기온(°C)', '최고기온(°C)', '평균 상대습도(%)', '합계 일조시간(hr)']

for col in cols_to_interpolate:
    if col in df.columns:
        # 지점별로 그룹을 나누어 앞뒤 값을 기준으로 중간값을 채움
        df[col] = df.groupby('지점명')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))

# (3) 보간 후에도 남은 결측치(데이터의 맨 처음이나 끝이 비어있는 경우)는 전체 평균 등으로 최소화
df = df.fillna(method='bfill').fillna(method='ffill')

# 4. 결과 저장 및 확인
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print("-" * 30)
print(f"결측치 보완 완료: {output_file}")
print("\n[남은 결측치 현황]")
print(df.isnull().sum())
print("-" * 30)

# 5. 보완 전후 비교 샘플 출력
print("보완된 데이터 상위 5행:")
print(df.head())