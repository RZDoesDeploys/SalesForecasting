import pandas as pd
import xgboost as xgb

def get_prediction(start_date, end_date):
    model_filename = "model/xgboost_model.json"
    reg = xgb.XGBRegressor()
    reg.load_model(model_filename)
    
    future = pd.date_range(start=start_date, end=end_date, freq='MS') 
    future_df = pd.DataFrame(index=future)

    future_df = create_features(future_df)

    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']

    future_df['pred'] = reg.predict(future_df[FEATURES])
    
    predictions_dict = future_df['pred'].to_dict()
    predictions_dict = {date.strftime('%Y-%m-%d'): pred for date, pred in predictions_dict.items()}
    
    return predictions_dict

def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df