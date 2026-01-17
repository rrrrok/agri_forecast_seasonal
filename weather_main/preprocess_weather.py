import pandas as pd
import os

# 파일 경로 설정
current_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_path, '지역별_기상.csv')
output_file = os.path.join(current_path, '지역별_기상_전처리완료.csv')

def read_csv_safe(path):
    encodings = ['utf-8', 'cp949', 'euc-kr', 'utf-8-sig']
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Failed to read {path} with encodings: {encodings}")

# 데이터 로드
try:
    df = read_csv_safe(input_file)
    print(f"파일 로드 성공: {input_file}")
    print(f"삭제 전 컬럼: {df.columns.tolist()}")
except Exception as e:
    print(f"파일 로드 실패: {e}")
    exit()

# 1. '지점' 컬럼 삭제
if '지점' in df.columns:
    df = df.drop(columns=['지점'])
    print("'지점' 컬럼 삭제 완료")
else:
    print("'지점' 컬럼이 존재하지 않아 삭제하지 않았습니다.")

# 2. 결측치 0으로 채우기
null_count = df.isnull().sum().sum()
df = df.fillna(0)
print(f"결측치 {null_count}개를 0으로 대체했습니다.")

# 결과 저장
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"저장 완료: {output_file}")
