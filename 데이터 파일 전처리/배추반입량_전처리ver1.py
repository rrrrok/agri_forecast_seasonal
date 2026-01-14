import pandas as pd

# 1. 파일 불러오기
# 파일 인코딩 문제 발생 시 encoding='cp949' 또는 'utf-8-sig'를 시도해보세요.
file_path = '배추반입량_전처리전.csv'
df = pd.read_csv(file_path)

# 2. 제외할 컬럼 리스트 정의
columns_to_drop = ['서울청과', '농협', '중앙청과', '동화청과', '한국청과', '대아청과']

# 데이터프레임에 해당 컬럼이 존재하는지 확인 후 삭제
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# 3. DATE 컬럼을 날짜 형식으로 변환 후 오름차순 정렬
if 'DATE' in df.columns:
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values(by='DATE', ascending=True)

# 4. 새로운 CSV 파일로 저장
output_path = '배추반입량_전처리ver1.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"파일 변환 완료: {output_path}")