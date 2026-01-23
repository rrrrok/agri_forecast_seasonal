import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_and_merge_data():
    """데이터 로드 및 기상 데이터 병합 (계절별 주산지 + 최적 누적 변수)"""
    # 1. 파일 로드
    df_price = pd.read_csv('가격데이터_순별.csv', encoding='utf-8-sig')
    df_daegwallyeong = pd.read_csv('대관령_순단위.csv', encoding='utf-8-sig')
    df_haenam = pd.read_csv('해남_순단위.csv', encoding='utf-8-sig')

    # 2. 날짜 파싱 (순 -> 일)
    def parse_date_kor(date_str):
        date_str = str(date_str)
        year = int(date_str[:4])
        month = int(date_str[4:6])
        period = date_str[6:]
        day = {'상순': 5, '중순': 15, '하순': 25}[period]
        return pd.Timestamp(year=year, month=month, day=day)

    df_price['date_obj'] = df_price['DATE'].apply(parse_date_kor)
    df_daegwallyeong['date_obj'] = df_daegwallyeong['일시'].apply(parse_date_kor)
    df_haenam['date_obj'] = df_haenam['일시'].apply(parse_date_kor)
    
    # 정렬
    df_price = df_price.sort_values('date_obj').reset_index(drop=True)

    # 3. 계절별 기상 데이터 매핑 
    # (12~5월: 해남, 6~11월: 대관령)
    weather_map_daegwallyeong = df_daegwallyeong.set_index('date_obj').to_dict('index')
    weather_map_haenam = df_haenam.set_index('date_obj').to_dict('index')
    
    # 최적 변수 및 누적 기간 설정
    # 변수명 | 누적기간 | 시차
    # 평균습도 | 22 순 | 12 순
    # 최저기온 | 9 순 | 0 순
    # 최고기온 | 25 순 | 8 순
    # 평균강수량 | 9 순 | 0 순
    # 평균기온 | 24 순 | 9 순
    # 평균일조시간 | 5 순 | 12 순
    # 평균일사량 | 30 순 | 9 순
    
    opt_features = [
        {'var': '평균습도', 'n': 22, 'lag': 12},
        {'var': '최저기온', 'n': 9, 'lag': 0},
        {'var': '최고기온', 'n': 25, 'lag': 8},
        {'var': '평균강수량', 'n': 9, 'lag': 0},
        {'var': '평균기온', 'n': 24, 'lag': 9},
        {'var': '평균일조시간', 'n': 5, 'lag': 12},
        {'var': '평균일사량', 'n': 30, 'lag': 9}
    ]
    
    # 필요한 컬럼만 추출하여 병합 리스트 생성
    weather_data_list = []
    
    for _, row in df_price.iterrows():
        curr_date = row['date_obj']
        month = curr_date.month
        
        if month in [12, 1, 2, 3, 4, 5]:
            data_map = weather_map_haenam
        else:
            data_map = weather_map_daegwallyeong
            
        if curr_date in data_map:
            weather_row = data_map[curr_date]
            
            # 최적 변수 추출
            extracted_data = {}
            for item in opt_features:
                col_name = f"{item['var']}_{item['n']}순"
                extracted_data[col_name] = weather_row.get(col_name, np.nan)
                
            weather_data_list.append(extracted_data)
        else:
            # 결측
            weather_data_list.append({f"{item['var']}_{item['n']}순": np.nan for item in opt_features})
            
    df_weather = pd.DataFrame(weather_data_list)
    df_merged = pd.concat([df_price, df_weather], axis=1)
    
    # 4. Lag 적용 (시차 반영)
    for item in opt_features:
        col_name = f"{item['var']}_{item['n']}순"
        lag = item['lag']
        if lag > 0:
            df_merged[f"{col_name}_lag{lag}"] = df_merged[col_name].shift(lag)
        else:
            df_merged[f"{col_name}_lag0"] = df_merged[col_name]
            
    # 결측치 제거 (Lag로 인한 결측 등)
    # 2018년 이전 데이터는 훈련에서 제외되지만 2018년 데이터가 있어야 2019년 예측 가능할 수 있음
    # 여기서는 단순 dropna
    df_merged = df_merged.dropna().reset_index(drop=True)
    
    return df_merged

def create_training_features(df):
    """추가 파생변수 생성 및 최종 학습 데이터셋 생성"""
    df = df.copy()
    
    # Target: 1순(7일) 후 가격
    df['target'] = df['평균가격'].shift(-1)
    
    # 기존 Baseline 파생변수 (가격 기반)
    df['price_lag_1'] = df['평균가격'].shift(1)
    df['price_lag_2'] = df['평균가격'].shift(2)
    df['price_ma_3'] = df['평균가격'].shift(1).rolling(3).mean()
    df['price_change'] = df['평균가격'].pct_change()
    
    # 계절성
    df['month'] = df['date_obj'].dt.month
    df['season_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['season_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 기상 변수 (이미 Lag 적용됨) - 모든 opt_feature 컬럼 사용
    # 이름 규칙: {var}_{n}순_lag{lag}
    
    return df.dropna()

def split_train_test(df):
    """2018-2024 학습, 2025 테스트"""
    train_df = df[df['date_obj'].dt.year < 2025].copy()
    test_df = df[df['date_obj'].dt.year == 2025].copy()
    return train_df, test_df

# 모델링 관련 함수들은 baseline과 유사 (생략 없이 구현)
def build_xgboost_model():
    return xgb.XGBRegressor(
        n_estimators=300, # 조금 늘림
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

def build_lstm_model(input_shape):
    model = keras.Sequential([
        layers.LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n[{model_name}] 성능:")
    print(f"  MAE:  {mae:,.0f}원")
    print(f"  RMSE: {rmse:,.0f}원")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

def plot_results(train_df, test_df, y_test, predictions, model_type):
    # 날짜 문자열 사용 (DATE 컬럼)
    dates = test_df['DATE'].values
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. 2025년 예측 비교
    ax = axes[0]
    ax.plot(dates, y_test, 'o-', label='실제 가격', linewidth=2)
    ax.plot(dates, predictions, 's--', label='예측 가격', linewidth=2)
    ax.set_title(f'2025년 배추 가격 예측 Result ({model_type})')
    ax.set_ylabel('가격(원)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. 오차
    ax = axes[1]
    errors = y_test - predictions
    ax.bar(dates, errors, color=['red' if e > 0 else 'blue' for e in errors])
    ax.set_title('예측 오차 (실제 - 예측)')
    ax.set_ylabel('오차(원)')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('price_prediction_v1_result.png')
    print("그래프 저장: price_prediction_v1_result.png")

def save_prediction_csv(test_df, y_test, predictions):
    result = pd.DataFrame({
        'DATE': test_df['DATE'],
        'Actual': y_test,
        'Predicted': predictions,
        'Error': y_test - predictions,
        'Error_Rate': ((y_test - predictions) / y_test) * 100
    })
    result.to_csv('prediction_results_v1.csv', index=False, encoding='utf-8-sig')
    print("결과 저장: prediction_results_v1.csv")

def main():
    print("="*70)
    print("배추 가격 예측 모델 Ver.1 (기상 변수 추가)")
    print("="*70)
    
    # 1. 데이터 준비
    df = load_and_merge_data()
    print(f"데이터 병합 완료: {len(df)}개")
    
    df = create_training_features(df)
    
    # 학습에 사용할 Feature 선정
    # 기존 가격 변수 + 기상 변수
    base_features = ['price_lag_1', 'price_lag_2', 'price_ma_3', 'price_change', 
                     'season_sin', 'season_cos']
    
    # 기상 변수 컬럼 자동 추출
    weather_features = [c for c in df.columns if '순_lag' in c]
    
    feature_cols = base_features + weather_features
    print(f"학습 Features ({len(feature_cols)}개):")
    print(feature_cols)
    
    # 2. 분할
    train_df, test_df = split_train_test(df)
    print(f"\n학습셋: {len(train_df)}개 (2018~2024)")
    print(f"테스트셋: {len(test_df)}개 (2025)")
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    # 3. Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # 4. XGBoost
    xgb_model = build_xgboost_model()
    xgb_model.fit(X_train_s, y_train)
    xgb_pred = xgb_model.predict(X_test_s)
    xgb_res = evaluate_model(y_test, xgb_pred, "XGBoost")
    
    # 5. LSTM
    X_train_lstm = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
    X_test_lstm = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
    
    lstm_model = build_lstm_model((1, X_train_s.shape[1]))
    es = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    lstm_model.fit(X_train_lstm, y_train_s, epochs=100, batch_size=16, 
                   validation_split=0.2, callbacks=[es], verbose=0)
    
    lstm_pred_s = lstm_model.predict(X_test_lstm, verbose=0)
    lstm_pred = scaler_y.inverse_transform(lstm_pred_s).flatten()
    lstm_res = evaluate_model(y_test, lstm_pred, "LSTM")
    
    # 6. 최종 모델 선택
    if xgb_res['mae'] < lstm_res['mae']:
        final_pred = xgb_pred
        model_name = "XGBoost"
        print("\n>> 최종 선택: XGBoost")
    else:
        final_pred = lstm_pred
        model_name = "LSTM"
        print("\n>> 최종 선택: LSTM")
        
    save_prediction_csv(test_df, y_test, final_pred)
    plot_results(train_df, test_df, y_test, final_pred, model_name)

if __name__ == "__main__":
    main()
