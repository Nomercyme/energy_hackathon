import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random

from math import ceil

def create_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure the DataFrame index is a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    
    # Extract various datetime features
    df['tm_d'] = df.index.day.astype(np.int8)
    df['tm_w'] = df.index.isocalendar().week.astype(np.int8)
    df['tm_m'] = df.index.month.astype(np.int8)
    df['tm_y'] = (df.index.year - df.index.year.min()).astype(np.int8)
    df['tm_wm'] = df['tm_d'].apply(lambda x: ceil(x / 7)).astype(np.int8)
    df['tm_dw'] = df.index.dayofweek.astype(np.int8)
    df['tm_w_end'] = (df['tm_dw'] >= 5).astype(np.int8)
    
    # Create features for the hour of the day and the half-hour of the day
    df['hour_of_day'] = df.index.hour.astype(np.int8)
    df['halfhour_of_day'] = (df.index.hour * 2 + df.index.minute // 30).astype(np.int8)
    
    # Create features for the EFA block of the day
    efa_block_start = 23  # EFA blocks start at 23:00
    efa_block_duration = 4  # EFA blocks last 4 hours
    efa_blocks_per_day = 24 // efa_block_duration
    
    df['efa_block'] = ((((df.index.hour - efa_block_start) % 24) // efa_block_duration)+1).astype(np.int8)
    
    return df