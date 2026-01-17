import pandas as pd
import os

# 파일 경로 설정
current_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_path, '지역별_기상_전처리본.csv')
output_file = os.path.join(current_path, '지역별_기상_최종_전처리본.csv')

# 데이터 로드
try:
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    print(f"파일 로드 성공: {input_file}")
    print(f"삭제 전 컬럼: {df.columns.tolist()}")
except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {input_file}")
    exit()

# 첫 번째 컬럼('지점') 삭제
# 이름으로 삭제하거나 인덱스로 삭제 가능 (여기서는 '지점' 이름 확인 후 삭제)
if '지점' in df.columns:
    df = df.drop(columns=['지점'])
    print("'지점' 컬럼 삭제 완료")
else:
    # 혹시 이름이 다를 경우를 대비해 첫 번째 컬럼 강제 삭제 (선택 사항)
    first_col = df.columns[0]
    df = df.drop(columns=[first_col])
    print(f"첫 번째 컬럼('{first_col}') 삭제 완료")

# 결과 저장
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"저장 완료: {output_file}")
print(f"최종 컬럼: {df.columns.tolist()}")
