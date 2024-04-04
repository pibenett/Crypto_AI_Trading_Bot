def simple_moving_average(df, window_size=25):
    return df['close'].rolling(window=window_size).mean()

def generate_signals(df):
    short_window = simple_moving_average(df, 25)
    long_window = simple_moving_average(df, 100)
    
    # Signal to buy when short MA crosses above long MA
    df['signal'] = 0
    df['signal'][short_window > long_window] = 1
    
    # Signal to sell when short MA crosses below long MA
    df['signal'][short_window < long_window] = -1
    return df
