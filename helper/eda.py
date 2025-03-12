import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def visualize_data(df_or_series, start_date=None, end_date=None, columns=None, is_price=False, ln_y=False):
    # Convert Series to DataFrame if necessary
    if isinstance(df_or_series, pd.Series):
        df = df_or_series.to_frame()
    else:
        df = df_or_series
    
    # Set default start and end dates if not provided
    if start_date is None:
        start_date = df.index.min()
    if end_date is None:
        end_date = df.index.max()
    
    # Filter the DataFrame by the date range
    df_filtered = df.loc[start_date:end_date]
    
    # Select the specified columns
    if columns is not None:
        df_filtered = df_filtered[columns]
    
    # Filter to only include float or integer columns
    df_filtered = df_filtered.select_dtypes(include=['float', 'int'])
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    for column in df_filtered.columns:
        plt.plot(df_filtered.index, df_filtered[column], label=column)

    plt.axhline(y = 0.0, color = 'r', linestyle = '-') 

    # Add legend
    plt.legend()
    
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Log scale the y-axis if specified
    if ln_y:
        plt.yscale('log')

    # Label the axes
    plt.xlabel('Date')
    plt.ylabel('£/MW/h' if is_price else None)
    
    # Show the plot
    plt.show()
    
# Example usage
# Assuming df is your DataFrame with a DateTime index
# visualize_data(df, start_date='2022-01-01', end_date='2022-12-31', columns=['Price', 'Volume'], is_price=True)