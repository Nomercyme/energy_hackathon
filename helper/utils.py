import pandas as pd

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # Replace special characters in column names with underscores
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    # Remove the substring "(£/MWh)" from column names
    df.columns = df.columns.str.replace('(£_MWh)', '', regex=False)
    return df

def naive_predictor(df: pd.DataFrame, target:str, shift:int)-> pd.DataFrame:
    # Shift the target column by one day to create the naive prediction
    naive_predictions = df[target].shift(shift)

    return naive_predictions