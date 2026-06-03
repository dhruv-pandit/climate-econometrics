import pandas as pd
import numpy as np

def round_lat_lon(df):
    df['latitude'] = df['latitude'].round(1)
    df['longitude'] = df['longitude'].round(1)
    return df

def square_columns(df, columns):
    for col in columns:
        df[f'{col}_squared'] = df[col] ** 2
    return df

def split_dfs_by_season(df, tourist_months, non_tourist_months):
    """
    Split extreme precipitation dataframes by tourist and non-tourist seasons.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data to be split by season
    tourist_months : list
        List of month numbers for tourist season
    non_tourist_months : list
        List of month numbers for non-tourist season
        
    Returns:
    --------

    df_tourist : pd.DataFrame
        DataFrame filtered for tourist months with appropriate suffixes
    df_nontourist : pd.DataFrame
        DataFrame filtered for non-tourist months with appropriate suffixes
    """

    

    df_tourist = df[df['month'].dt.month.isin(tourist_months)].copy()
    df_tourist = df_tourist.rename(columns=lambda x: f"{x}_m_t" if x not in ['latitude', 'longitude', 'month'] else x)
        
    # Filter for non-tourist months
    df_nontourist = df[df['month'].dt.month.isin(non_tourist_months)].copy()
    df_nontourist = df_nontourist.rename(columns=lambda x: f"{x}_m_nt" if x not in ['latitude', 'longitude', 'month'] else x)
    
    return df_tourist, df_nontourist


def calculate_daily_threshold_values(df_daily, pct_filepath, value_col='tmean', baseline='61_90', agg='count'):
    """
    Calculates conditions above/below the respective monthly percentiles.
    Takes a dataframe `df_daily` that already contains daily totals or means.
    agg='count': Returns 1 if condition met for that day, 0 otherwise.
    agg='sum': Returns the actual daily value if condition met, 0 otherwise (used for precipitation accumulation).
    """
    pct_df = pd.read_parquet(pct_filepath)
    pct_df = round_lat_lon(pct_df)
    
    # 1. Ensure we have a date column
    df = df_daily.copy()
    if 'date' not in df.columns:
        df['date'] = df['valid_time'].dt.normalize()
    
    # 2. Extract month to merge with monthly percentiles
    df['month'] = df['date'].dt.month
    
    # 3. Merge with the percentile thresholds
    df = df.merge(
        pct_df[['latitude', 'longitude', 'month', 'p5', 'p10', 'p90', 'p95', 'p99']],
        on=['latitude', 'longitude', 'month'],
        how='left'
    )
    
    value = df[value_col]
    
    if agg == 'count':
        df[f'above_p90_{baseline}'] = (value > df['p90']).astype('int8')
        df[f'above_p95_{baseline}'] = (value > df['p95']).astype('int8')
        df[f'above_p99_{baseline}'] = (value > df['p99']).astype('int8')
        df[f'below_p5_{baseline}']  = (value < df['p5']).astype('int8')
        df[f'below_p10_{baseline}'] = (value < df['p10']).astype('int8')
    elif agg == 'sum':
        df[f'above_p90_{baseline}'] = np.where(value > df['p90'], value, 0)
        df[f'above_p95_{baseline}'] = np.where(value > df['p95'], value, 0)
        df[f'above_p99_{baseline}'] = np.where(value > df['p99'], value, 0)
        df[f'below_p5_{baseline}']  = np.where(value < df['p5'],  value, 0)
        df[f'below_p10_{baseline}'] = np.where(value < df['p10'], value, 0)
    else:
        raise ValueError(f"agg must be 'count' or 'sum', got '{agg}'")
    
    # 5. Filter only the relevant columns to return
    keep_cols = [
        'latitude', 'longitude', 'date', 
        f'above_p90_{baseline}', f'above_p95_{baseline}', f'above_p99_{baseline}',
        f'below_p5_{baseline}', f'below_p10_{baseline}'
    ]
    
    return df[keep_cols]