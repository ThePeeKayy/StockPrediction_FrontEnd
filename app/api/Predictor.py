from flask import Flask, request, jsonify
import torch
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytorch_forecasting import TimeSeriesDataSet, NBeats, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import GroupNormalizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import requests
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

def load_scalers(symbol):
    try:
        with open(f'./Scalers/{symbol}_feature_scaler.pkl', 'rb') as f:
            feature_scaler = pickle.load(f)
        with open(f'./Scalers/{symbol}_value_scaler.pkl', 'rb') as f:
            value_scaler = pickle.load(f)
            
        return feature_scaler, value_scaler
    except FileNotFoundError as e:
        print(f"Scaler file not found: {e}")
        return None, None, None

def extrapolate_economic_data(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    current_date = datetime.now()
    next_month = pd.Timestamp(current_date.year, current_date.month, 1) + pd.DateOffset(months=1)
    
    for col in ['GDP', 'CPIAUCSL']:
        if col in df.columns:
            series = df.set_index('date')[col].dropna()
            if len(series) > 2:
                last_date = series.index[-1]
                if last_date < next_month:
                    if col == 'GDP':
                        quarterly_data = {}
                        for date, value in series.items():
                            quarter_start = pd.Timestamp(date.year, ((date.month - 1) // 3) * 3 + 1, 1)
                            quarterly_data[quarter_start] = value
                        
                        last_quarter = max(quarterly_data.keys())
                        last_quarter_value = quarterly_data[last_quarter]
                        
                        lookback_quarters = min(4, len(quarterly_data))
                        recent_quarters = list(quarterly_data.items())[-lookback_quarters:]
                        
                        if len(recent_quarters) > 1:
                            x = np.arange(len(recent_quarters))
                            y = [v for _, v in recent_quarters]
                            slope, intercept = np.polyfit(x, y, 1)
                            next_quarter_value = last_quarter_value + slope
                        else:
                            next_quarter_value = last_quarter_value * 1.005
                        
                        quarter_after_last = last_quarter + pd.DateOffset(months=3)
                        
                        date_range = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                                 end=next_month, freq='MS')
                        
                        for future_date in date_range:
                            if future_date < quarter_after_last:
                                predicted_value = last_quarter_value
                            else:
                                predicted_value = next_quarter_value
                            
                            df_idx = df[df['date'] == future_date].index
                            if len(df_idx) > 0:
                                df.loc[df_idx[0], col] = predicted_value
                            else:
                                new_row = df.iloc[[-1]].copy()
                                new_row['date'] = future_date
                                new_row[col] = predicted_value
                                for other_col in df.columns:
                                    if other_col not in ['date', col]:
                                        new_row[other_col] = df[other_col].iloc[-1]
                                df = pd.concat([df, new_row], ignore_index=True)
                    
                    else:  
                        lookback = min(6, len(series))
                        recent_series = series.iloc[-lookback:]
                        
                        x = np.arange(len(recent_series))
                        y = recent_series.values
                        slope, intercept = np.polyfit(x, y, 1)
                        
                        date_range = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                                 end=next_month, freq='MS')
                        
                        last_value = series.iloc[-1]
                        
                        for i, future_date in enumerate(date_range, 1):
                            predicted_value = last_value + slope * i
                            df_idx = df[df['date'] == future_date].index
                            if len(df_idx) > 0:
                                df.loc[df_idx[0], col] = predicted_value
                            else:
                                new_row = df.iloc[[-1]].copy()
                                new_row['date'] = future_date
                                new_row[col] = predicted_value
                                for other_col in df.columns:
                                    if other_col not in ['date', col]:
                                        new_row[other_col] = df[other_col].iloc[-1]
                                df = pd.concat([df, new_row], ignore_index=True)
    
    df = df.sort_values('date').reset_index(drop=True)
    df = df.drop_duplicates(subset=['date'], keep='last')
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df.fillna(method='ffill').fillna(method='bfill')

def fetch_data(fred_key):
    import yfinance as yf
    
    def fetch_fred_data(series_id, start_date, end_date):
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            'series_id': series_id,
            'api_key': fred_key,
            'file_type': 'json',
            'observation_start': start_date.strftime('%Y-%m-%d'),
            'observation_end': end_date.strftime('%Y-%m-%d')
        }
        response = requests.get(url, params=params, timeout=10).json()
        df = pd.DataFrame(response['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df['value'].dropna().rename(series_id)

    def fetch_yfinance_data(symbol):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=18*30)
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        data_resampled = data['Close'].resample('MS').first()
        return data_resampled.rename(symbol)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=18*30)
    fred_series_ids = ['GDP', 'CPIAUCSL']
    yfinance_symbols = ['SPY', 'DIA', 'QQQ']
    combined_data = pd.DataFrame()

    for series_id in fred_series_ids:
        fred_series = fetch_fred_data(series_id, start_date, end_date)
        combined_data = pd.concat([combined_data, fred_series], axis=1)

    for symbol in yfinance_symbols:
        yf_series = fetch_yfinance_data(symbol)
        combined_data = pd.concat([combined_data, yf_series], axis=1)
    
    combined_data.reset_index(inplace=True)
    combined_data.rename(columns={'index': 'date'}, inplace=True)
    combined_data['date'] = pd.to_datetime(combined_data['date'], utc=True).dt.tz_localize(None)
    combined_data['date'] = combined_data['date'].dt.strftime('%Y-%m-%d')
    combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
    return combined_data

def compute_indicators(df):
    df = df.copy()
    df['SMA_20'] = df['value'].rolling(window=20, min_periods=1).mean()
    df['EMA_20'] = df['value'].ewm(span=20, adjust=False).mean()
    df['SMA_5'] = df['value'].rolling(window=5, min_periods=1).mean()
    
    delta = df['value'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    df['Volatility_10'] = df['value'].pct_change().rolling(10, min_periods=1).std()
    df['Volatility_10'] = df['Volatility_10'].fillna(df['Volatility_10'].mean())
    
    exp1 = df['value'].ewm(span=12, adjust=False).mean()
    exp2 = df['value'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['Volume_MA_10'] = df['Volume'].rolling(window=10, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA_10'] + 1e-8)
        df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1.0)
    else:
        df['Volume_Ratio'] = 1.0
    
    for col in ['SMA_20', 'EMA_20', 'SMA_5', 'MACD', 'MACD_Signal']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    
    return df

def transform_polygon_data(raw_data):
    df = pd.DataFrame(raw_data)
    column_mapping = {
        't': 'date', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'value', 'v': 'Volume'
    }
    df = df.rename(columns=column_mapping)
    df['date'] = pd.to_datetime(df['date'], unit='ms').dt.normalize()
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    required_columns = ['date', 'Open', 'High', 'Low', 'value', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    return df[required_columns].to_dict('records')

def prepare_data_for_prediction(current_data, economic_data, symbol):
    current_data_df = pd.DataFrame(current_data)
    economic_data_df = pd.DataFrame(economic_data)
    
    if 'Date' in current_data_df.columns:
        current_data_df.rename(columns={'Date': 'date', 'Close': 'value'}, inplace=True)
    
    current_data_df['date'] = pd.to_datetime(current_data_df['date']).dt.normalize()
    economic_data_df['date'] = pd.to_datetime(economic_data_df['date']).dt.normalize()
    current_data_df = current_data_df.sort_values('date').reset_index(drop=True)
    economic_data_df = economic_data_df.sort_values('date').reset_index(drop=True)
    
    merged_data = current_data_df.copy()
    
    if 'GDP' in economic_data_df.columns:
        gdp_series = economic_data_df[['date', 'GDP']].copy()
        gdp_series['GDP'] = gdp_series['GDP'] / 50
        gdp_series.rename(columns={'GDP': 'gdp'}, inplace=True)
        merged_data = pd.merge_asof(merged_data, gdp_series, on='date', direction='backward')
    
    if 'CPIAUCSL' in economic_data_df.columns:
        cpi_series = economic_data_df[['date', 'CPIAUCSL']].copy()
        cpi_series.rename(columns={'CPIAUCSL': 'cpiaucsl'}, inplace=True)
        merged_data = pd.merge_asof(merged_data, cpi_series, on='date', direction='backward')
    
    if 'SPY' in economic_data_df.columns:
        spy_series = economic_data_df[['date', 'SPY']].copy()
        spy_series.rename(columns={'SPY': 'spy'}, inplace=True)
        merged_data = pd.merge_asof(merged_data, spy_series, on='date', direction='backward')
    
    merged_data['month'] = merged_data['date'].dt.month.astype(str)
    merged_data['day_of_week'] = merged_data['date'].dt.dayofweek
    merged_data['group'] = '0'
    merged_data = compute_indicators(merged_data)
    merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')
    merged_data = merged_data.reset_index(drop=True)
    merged_data['time_idx'] = np.arange(len(merged_data))
    return merged_data

def predict_tft(current_data, economic_data, symbol):
    merged_data = prepare_data_for_prediction(current_data, economic_data, symbol)
    feature_columns = ['SMA_5', 'SMA_20', 'EMA_20', 'RSI', 'Volatility_10', 
                      'MACD', 'MACD_Signal', 'Volume_Ratio']
    
    
    feature_scaler, value_scaler = load_scalers(symbol)
    if feature_scaler is None or value_scaler is None:
        print(f"Error: Could not load scalers for {symbol}")
        return np.zeros(30)  
    
    
    merged_data[feature_columns] = feature_scaler.transform(merged_data[feature_columns])
    merged_data['value'] = value_scaler.transform(merged_data[['value']])
    
    max_encoder_length = 90
    max_prediction_length = 30
    
    training_dataset = TimeSeriesDataSet(
        merged_data,
        time_idx='time_idx',
        target='value',
        group_ids=['group'],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        time_varying_known_categoricals=['month'],
        time_varying_known_reals=['time_idx', 'spy', 'gdp'] if 'spy' in merged_data.columns else ['time_idx'],
        time_varying_unknown_reals=feature_columns,
        target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus", center=False),
        add_relative_time_idx=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    
    
    model_path = f"./TFTData/epoch=49-val_MAE=0.2111-{symbol}.ckpt"
    if not os.path.exists(model_path):
        
        model_path = "./TFTData/epoch=49-val_MAE=0.2111-AAPL.ckpt"
        print(f"Warning: Using default model path {model_path}")
    
    model = TemporalFusionTransformer.load_from_checkpoint(model_path)
    
    test_encoder_data = merged_data.iloc[-max_encoder_length:].copy().reset_index(drop=True)
    test_decoder_data = test_encoder_data.iloc[-max_prediction_length:].copy()
    last_time_idx = test_encoder_data['time_idx'].max()
    test_decoder_data['time_idx'] = np.arange(last_time_idx + 1, last_time_idx + 1 + max_prediction_length)
    last_date = test_encoder_data['date'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=max_prediction_length + 1, freq='D')[1:]
    test_decoder_data['date'] = future_dates
    test_decoder_data['month'] = test_decoder_data['date'].dt.month.astype(str)
    test_decoder_data['day_of_week'] = test_decoder_data['date'].dt.dayofweek
    
    for col in ['spy', 'gdp', 'cpiaucsl']:
        if col in test_encoder_data.columns:
            test_decoder_data[col] = test_encoder_data[col].iloc[-1]
    
    prediction_data = pd.concat([test_encoder_data, test_decoder_data], ignore_index=True)
    prediction_dataset = TimeSeriesDataSet.from_dataset(training_dataset, prediction_data, stop_randomization=True)
    prediction_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    
    raw_predictions = model.predict(prediction_dataloader, mode='raw')
    num_quantiles = raw_predictions['prediction'].shape[2]
    middle_quantile_index = num_quantiles // 2
    predictions = raw_predictions['prediction'][:, :, middle_quantile_index].detach().cpu().numpy()
    selected_batch_predictions = predictions[-1]
    
    
    predicted_values = value_scaler.inverse_transform(selected_batch_predictions.reshape(-1, 1)).flatten()
    return predicted_values

def predict_nbeats(current_data, symbol):
    df = pd.DataFrame(current_data)
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.sort_values('date').reset_index(drop=True)
    
 
    df = df.reset_index(drop=True)
    df['time_idx'] = np.arange(len(df))
    df['group'] = '0'

    max_encoder_length = 90
    max_prediction_length = 30

    dataset = TimeSeriesDataSet(
        df,
        time_idx='time_idx',
        target='value',
        group_ids=['group'],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=['value'],
        target_normalizer=GroupNormalizer(groups=["group"], transformation=None, center=False),
        add_relative_time_idx=False,
    )
    
    
    model_path = f"./NbeatsData/epoch=41-val_loss=0.0157-{symbol}.ckpt"
    if not os.path.exists(model_path):
        
        model_path = "./NbeatsData/epoch=41-val_loss=0.0157-AAPL.ckpt"
        print(f"Warning: Using default model path {model_path}")
    
    model = NBeats.load_from_checkpoint(model_path)
    forecast_dataset = TimeSeriesDataSet.from_dataset(dataset, df.iloc[-120:], predict=True, stop_randomization=True)
    forecast_dataloader = forecast_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
    forecasts = model.predict(forecast_dataloader)
    forecasts_np = forecasts.cpu().numpy()
    last_forecast = forecasts_np[-1]
    
    
    predicted_values = last_forecast.reshape(-1, 1).flatten()
    return predicted_values

@app.route('/predictor', methods=['POST'])
def predict():
    data = request.json
    symbol = data.get('symbol', 'AAPL').upper()
    current_data = data.get('current_data')
    fred_key = data.get('fred_key')
    
    current_data = transform_polygon_data(current_data)
    combined_data = fetch_data(fred_key)
    combined_data_df = pd.DataFrame(combined_data)
    
    combined_data_df = extrapolate_economic_data(combined_data_df)
    
    tft_pred = predict_tft(current_data, combined_data_df, symbol)
    nbeats_pred = predict_nbeats(current_data, symbol)
    
    historical_data = pd.DataFrame(current_data)
    historical_data['model'] = 'historical'
    last_date = pd.to_datetime(pd.DataFrame(current_data)['date'].max())
    tft_dates = pd.date_range(start=last_date, periods=len(tft_pred) + 1, freq='D')[1:]
    nbeats_dates = pd.date_range(start=last_date, periods=len(nbeats_pred) + 1, freq='D')[1:]
    
    all_data = []
    
    for _, row in historical_data.iterrows():
        all_data.append({
            'date': row['date'],
            'value': float(row['value']),
            'model': 'historical'
        })
    
    for i, pred in enumerate(tft_pred):
        all_data.append({
            'date': tft_dates[i].strftime('%Y-%m-%d'),
            'value': float(pred),
            'model': 'tft'
        })
    
    for i, pred in enumerate(nbeats_pred):
        all_data.append({
            'date': nbeats_dates[i].strftime('%Y-%m-%d'),
            'value': float(pred),
            'model': 'nbeats'
        })
    
    economic_data_json = combined_data_df.to_json(orient='records')
    prediction_data_json = pd.DataFrame(all_data).to_json(orient='records')
    return jsonify([economic_data_json, prediction_data_json])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)