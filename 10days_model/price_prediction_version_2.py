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
    """데이터 로드 및 기상 데이터 병합 (계절별 주산지 + Version 2 설정)"""
    # 1. 파일 로드
    df_price = pd.read_csv('가격데이터_순별.csv', encoding='utf-8-sig')
    df_daegwallyeong = pd.read_csv('대관령_순단위.csv', encoding='utf-8-sig')
    df_haenam = pd.read_csv('해남_순단위.csv', encoding='utf-8-sig')

    # 2. 날짜 파싱
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
    
    df_price = df_price.sort_values('date_obj').reset_index(drop=True)

    # 3. 계절별 기상 데이터 매핑
    weather_map_daegwallyeong = df_daegwallyeong.set_index('date_obj').to_dict('index')
    weather_map_haenam = df_haenam.set_index('date_obj').to_dict('index')
    
    # Version 2 설정: 누적 3순, 시차 3순
    target_n = 3
    target_lag = 3
    weather_vars = ['평균기온', '최고기온', '최저기온', '평균강수량', '평균습도', '평균일조시간', '평균일사량']
    
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
            extracted_data = {}
            for var in weather_vars:
                col_name = f"{var}_{target_n}순"
                extracted_data[col_name] = weather_row.get(col_name, np.nan)
            weather_data_list.append(extracted_data)
        else:
            weather_data_list.append({f"{var}_{target_n}순": np.nan for var in weather_vars})
            
    df_weather = pd.DataFrame(weather_data_list)
    df_merged = pd.concat([df_price, df_weather], axis=1)
    
    # 4. Lag 적용
    for var in weather_vars:
        col_name = f"{var}_{target_n}순"
        if target_lag > 0:
            df_merged[f"{col_name}_lag{target_lag}"] = df_merged[col_name].shift(target_lag)
        else:
            df_merged[f"{col_name}_lag0"] = df_merged[col_name]
            
    df_merged = df_merged.dropna().reset_index(drop=True)
    return df_merged

def create_training_features(df):
    df = df.copy()
    df['target'] = df['평균가격'].shift(-1)
    
    df['price_lag_1'] = df['평균가격'].shift(1)
    df['price_lag_2'] = df['평균가격'].shift(2)
    df['price_ma_3'] = df['평균가격'].shift(1).rolling(3).mean()
    df['price_change'] = df['평균가격'].pct_change()
    
    df['month'] = df['date_obj'].dt.month
    df['season_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['season_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df.dropna()

def split_train_test(df):
    train_df = df[df['date_obj'].dt.year < 2025].copy()
    test_df = df[df['date_obj'].dt.year == 2025].copy()
    return train_df, test_df

def build_xgboost_model():
    return xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42
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
    dates = test_df['DATE'].values
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    ax = axes[0]
    ax.plot(dates, y_test, 'o-', label='실제 가격', linewidth=2)
    ax.plot(dates, predictions, 's--', label='예측 가격', linewidth=2)
    ax.set_title(f'2025년 가격 예측 (Ver.2 - 누적3순/시차3순) - {model_type}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    ax = axes[1]
    errors = y_test - predictions
    ax.bar(dates, errors, color=['red' if e > 0 else 'blue' for e in errors])
    ax.set_title('예측 오차')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('price_prediction_v2_result.png')
    print("그래프 저장: price_prediction_v2_result.png")

def save_prediction_csv(test_df, y_test, predictions):
    result = pd.DataFrame({
        'DATE': test_df['DATE'],
        'Actual': y_test,
        'Predicted': predictions,
        'Error': y_test - predictions
    })
    result.to_csv('prediction_results_v2.csv', index=False, encoding='utf-8-sig')
    print("결과 저장: prediction_results_v2.csv")

def main():
    print("="*70)
    print("배추 가격 예측 모델 Ver.2 (누적 3순, 시차 3순)")
    print("="*70)
    
    df = load_and_merge_data()
    df = create_training_features(df)
    
    base_features = ['price_lag_1', 'price_lag_2', 'price_ma_3', 'price_change', 
                     'season_sin', 'season_cos']
    weather_features = [c for c in df.columns if '순_lag' in c]
    feature_cols = base_features + weather_features
    print(f"학습 변수: {feature_cols}")
    
    train_df, test_df = split_train_test(df)
    
    X_train = train_df[feature_cols].values; y_train = train_df['target'].values
    X_test = test_df[feature_cols].values; y_test = test_df['target'].values
    
    scaler_X = StandardScaler(); scaler_y = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # XGBoost
    xgb_model = build_xgboost_model()
    xgb_model.fit(X_train_s, y_train)
    xgb_pred = xgb_model.predict(X_test_s)
    xgb_res = evaluate_model(y_test, xgb_pred, "XGBoost")
    
    # LSTM
    X_train_lstm = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
    X_test_lstm = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
    lstm_model = build_lstm_model((1, X_train_s.shape[1]))
    es = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    lstm_model.fit(X_train_lstm, y_train_s, epochs=100, batch_size=16, validation_split=0.2, callbacks=[es], verbose=0)
    lstm_pred = scaler_y.inverse_transform(lstm_model.predict(X_test_lstm, verbose=0)).flatten()
    lstm_res = evaluate_model(y_test, lstm_pred, "LSTM")
    
    if xgb_res['mae'] < lstm_res['mae']:
        final_pred = xgb_pred; model_name = "XGBoost"
        print("\n>> 최종 선택: XGBoost")
    else:
        final_pred = lstm_pred; model_name = "LSTM"
        print("\n>> 최종 선택: LSTM")
        
    save_prediction_csv(test_df, y_test, final_pred)
    plot_results(train_df, test_df, y_test, final_pred, model_name)

if __name__ == "__main__":
    main()
