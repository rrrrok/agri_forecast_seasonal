import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_and_preprocess_data(file_path):
    """CSV 파일 로드 및 전처리"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    def parse_date(date_str):
        year, month, period = date_str[:4], date_str[4:6], date_str[6:]
        day = {'상순': '05', '중순': '15', '하순': '25'}[period]
        return pd.to_datetime(f"{year}-{month}-{day}")
    
    df['date'] = df['DATE'].apply(parse_date)
    df = df.sort_values('date').reset_index(drop=True)
    df['전년'] = df['전년'].fillna(df['평균가격'])
    df['평년'] = df['평년'].fillna(df['평균가격'])
    
    return df


def create_features(df, lookback=3, forecast_horizon=1):
    """시계열 피처 생성 (데이터 누수 방지)"""
    df = df.copy()
    
    # Target: 7일(1순) 후 가격
    df['target'] = df['평균가격'].shift(-forecast_horizon)
    
    # Lag features
    for i in range(1, lookback + 1):
        df[f'price_lag_{i}'] = df['평균가격'].shift(i)
    
    # Moving averages
    df['price_ma_3'] = df['평균가격'].shift(1).rolling(3, min_periods=1).mean()
    df['price_ma_6'] = df['평균가격'].shift(1).rolling(6, min_periods=1).mean()
    
    # Price changes
    df['price_change'] = df['평균가격'].pct_change()
    df['price_change_lag_1'] = df['price_change'].shift(1)
    
    # Ratios
    df['ratio_to_prev_year'] = df['평균가격'] / (df['전년'] + 1)
    df['ratio_to_normal'] = df['평균가격'] / (df['평년'] + 1)
    
    # Seasonality
    df['month'] = df['date'].dt.month
    df['season_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['season_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Additional features
    df['prev_year'] = df['전년']
    df['normal_price'] = df['평년']
    
    return df


def split_train_test(df):
    """데이터 분할: 2018-2024 학습, 2025 테스트"""
    df_clean = df.dropna(subset=['target']).copy()
    train_df = df_clean[df_clean['date'].dt.year < 2025].copy()
    test_df = df_clean[df_clean['date'].dt.year == 2025].copy()
    return train_df, test_df


def build_lstm_model(input_shape):
    """LSTM 모델"""
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


def build_xgboost_model():
    """XGBoost 모델"""
    return xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )


def evaluate_model(y_true, y_pred, model_name):
    """모델 성능 평가"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{model_name} 성능:")
    print(f"  MAE:  {mae:,.0f}원")
    print(f"  RMSE: {rmse:,.0f}원")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}


def plot_results(train_df, test_df, y_test, predictions, model_type):
    """결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 전체 데이터
    ax = axes[0, 0]
    ax.plot(train_df['date'], train_df['평균가격'], label='학습 데이터', alpha=0.7)
    ax.plot(test_df['date'], test_df['평균가격'], label='테스트 현재가격', marker='o', alpha=0.7)
    ax.plot(test_df['date'], y_test, label='실제 (7일후)', marker='o', linewidth=2)
    ax.plot(test_df['date'], predictions, label=f'예측 ({model_type})', marker='s', linewidth=2)
    ax.set_xlabel('날짜')
    ax.set_ylabel('가격 (원)')
    ax.set_title('배추 가격 예측 (7일 후)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. 2025년 확대
    ax = axes[0, 1]
    ax.plot(test_df['DATE'], y_test, label='실제', marker='o', linewidth=2)
    ax.plot(test_df['DATE'], predictions, label='예측', marker='s', linewidth=2)
    ax.set_xlabel('날짜')
    ax.set_ylabel('가격 (원)')
    ax.set_title('2025년 예측 상세')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. 오차 분포
    ax = axes[1, 0]
    errors = y_test - predictions
    colors = ['red' if e > 0 else 'blue' for e in errors]
    ax.bar(range(len(errors)), errors, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('테스트 샘플')
    ax.set_ylabel('오차 (원)')
    ax.set_title('예측 오차 분포')
    ax.grid(True, alpha=0.3)
    
    # 4. 산점도
    ax = axes[1, 1]
    ax.scatter(y_test, predictions, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('실제값 (원)')
    ax.set_ylabel('예측값 (원)')
    ax.set_title('실제 vs 예측')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cabbage_price_prediction.png', dpi=300, bbox_inches='tight')
    print("\n그래프 저장: cabbage_price_prediction.png")


def save_results(test_df, y_test, predictions):
    """예측 결과 저장"""
    result_df = pd.DataFrame({
        '날짜': test_df['DATE'].values,
        '현재가격': test_df['평균가격'].values,
        '실제_7일후': y_test,
        '예측_7일후': predictions,
        '오차': y_test - predictions,
        '오차율(%)': ((y_test - predictions) / y_test) * 100
    })
    
    result_df.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
    print("결과 저장: prediction_results.csv\n")
    print("="*70)
    print("예측 결과")
    print("="*70)
    print(result_df.to_string(index=False))


def main(file_path):
    """메인 실행 함수"""
    print("="*70)
    print("배추 가격 예측 모델 (7일 후 예측)")
    print("="*70)
    
    # 데이터 로드
    df = load_and_preprocess_data(file_path)
    print(f"\n전체 데이터: {len(df)}개 ({df['date'].min().date()} ~ {df['date'].max().date()})")
    
    # 피처 생성
    df = create_features(df, lookback=3, forecast_horizon=1)
    
    # 데이터 분할
    train_df, test_df = split_train_test(df)
    print(f"학습 데이터: {len(train_df)}개 ({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")
    print(f"테스트 데이터: {len(test_df)}개 ({test_df['date'].min().date()} ~ {test_df['date'].max().date()})")
    
    # 피처 선택
    feature_cols = [
        'price_lag_1', 'price_lag_2', 'price_lag_3',
        'price_ma_3', 'price_ma_6', 'price_change_lag_1',
        'ratio_to_prev_year', 'ratio_to_normal',
        'season_sin', 'season_cos', 'prev_year', 'normal_price'
    ]
    
    train_df = train_df.dropna(subset=feature_cols).reset_index(drop=True)
    test_df = test_df.dropna(subset=feature_cols).reset_index(drop=True)
    
    X_train, y_train = train_df[feature_cols].values, train_df['target'].values
    X_test, y_test = test_df[feature_cols].values, test_df['target'].values
    
    # 정규화
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # XGBoost 학습
    print("\n" + "="*70)
    print("XGBoost 학습")
    print("="*70)
    xgb_model = build_xgboost_model()
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_metrics = evaluate_model(y_test, xgb_pred, "XGBoost")
    
    # LSTM 학습
    print("\n" + "="*70)
    print("LSTM 학습")
    print("="*70)
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    lstm_model = build_lstm_model((1, X_train_scaled.shape[1]))
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lstm_model.fit(X_train_lstm, y_train_scaled, epochs=100, batch_size=16, 
                   validation_split=0.2, callbacks=[early_stop], verbose=0)
    
    lstm_pred_scaled = lstm_model.predict(X_test_lstm, verbose=0)
    lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled).flatten()
    lstm_metrics = evaluate_model(y_test, lstm_pred, "LSTM")
    
    # 최적 모델 선택
    print("\n" + "="*70)
    if xgb_metrics['mae'] < lstm_metrics['mae']:
        print("✓ 최종 모델: XGBoost (MAE 기준)")
        final_predictions, model_type = xgb_pred, 'XGBoost'
    else:
        print("✓ 최종 모델: LSTM (MAE 기준)")
        final_predictions, model_type = lstm_pred, 'LSTM'
    print("="*70)
    
    # 시각화 및 결과 저장
    plot_results(train_df, test_df, y_test, final_predictions, model_type)
    save_results(test_df, y_test, final_predictions)
    
    return xgb_model if model_type == 'XGBoost' else lstm_model, scaler_X, scaler_y, model_type


if __name__ == "__main__":
    file_path = "가격데이터_순별.csv"
    model, scaler_X, scaler_y, model_type = main(file_path)
    print(f"\n{'='*70}")
    print(f"학습 완료 - 최종 모델: {model_type}")
    print(f"{'='*70}")