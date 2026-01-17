import pandas as pd
import os

# 1. 파일 경로 설정 (사용자 환경에 맞게 수정)
# 현재 실행 파일 위치를 기준으로 경로 설정
current_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_path, '지역별_기상_전처리완료.csv')
output_file = os.path.join(current_path, '지역별_기상_누적_추가.csv')

# 2. 데이터 로드
print("데이터를 불러오는 중입니다...")
df = pd.read_csv(input_file)
df['일시'] = pd.to_datetime(df['일시'])

# 3. 누적 변수 생성 설정
# 농산물 생육에 중요한 기간들 (7일, 14일, 30일, 60일, 90일, 120일)
windows = [7, 14, 30, 60, 90, 120]

# 누적 계산을 할 대상 변수들
# 평균기온 -> 누적하면 '적산온도(GDD)' 개념이 됨 (작물 성숙도와 직결)
target_cols = {
    '평균기온(°C)': '적산온도',
    '일강수량(mm)': '누적강수량',
    '합계 일조시간(hr)': '누적일조시간',
    '합계 일사량(MJ/m2)': '누적일사량'
}

# 4. 지역별 그룹화 및 누적 계산 (핵심!)
# 지역이 섞이지 않게 '지점명'으로 묶은 뒤 rolling을 적용해야 함
print("누적 변수를 생성 중입니다")

for col_name, new_name in target_cols.items():
    # 해당 컬럼이 데이터에 있는지 확인
    if col_name in df.columns:
        for w in windows:
            # 변수명 예시: 적산온도_60d, 누적강수량_30d
            new_col = f'{new_name}_{w}d'
            
            # GroupBy 후 Rolling Sum 적용
            df[new_col] = df.groupby('지점명')[col_name].transform(lambda x: x.rolling(window=w, min_periods=1).sum())
            
            print(f"✅ 생성 완료: {new_col}")

# 5. 결과 저장
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n[완료] 파일이 생성되었습니다: {output_file}")
print("이제 이 파일을 모델 학습에 사용하시면 성능이 크게 향상될 것입니다.")