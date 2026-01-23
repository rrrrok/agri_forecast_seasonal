import pandas as pd
import os

# 파일 경로 설정
current_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(current_path, '지역별_기상_누적_추가.csv')

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
    print(f"전체 데이터 행 수: {len(df)}")
except Exception as e:
    print(f"파일 로드 실패: {e}")
    exit()

# 지역별로 분리하여 저장
if '지점명' in df.columns:
    regions = df['지점명'].unique()
    print(f"발견된 지역: {regions}")

    for region in regions:
        region_df = df[df['지점명'] == region]
        # 파일명에 특수문자나 공백이 있을 수 있으므로 처리 (단순화)
        clean_region_name = str(region).replace(' ', '_').replace('/', '_')
        output_file = os.path.join(current_path, f'{clean_region_name}_기상.csv')
        
        region_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"저장 완료: {output_file} ({len(region_df)}행)")
else:
    print("'지점명' 컬럼을 찾을 수 없어 분리할 수 없습니다.")
    print(f"현재 컬럼: {df.columns.tolist()}")
