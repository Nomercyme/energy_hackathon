import pandas as pd

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # Replace special characters in column names with underscores
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    # Remove the substring "(£/MWh)" from column names
    df.columns = df.columns.str.replace('(£_MWh)', '', regex=False)
    return df