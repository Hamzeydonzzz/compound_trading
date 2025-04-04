import talib
import numpy as np

def verify_talib():
    """
    Verify TA-Lib installation and basic functionality
    """
    print("TA-Lib Version:", talib.__version__)
    
    # Generate some random price data
    close_prices = np.random.random(100) * 100  # 100 random prices between 0-100
    
    # Calculate Simple Moving Average (SMA)
    sma_10 = talib.SMA(close_prices, timeperiod=10)
    print("\nSimple Moving Average (10-period) Test:")
    print("Input prices (first 10):", close_prices[:10])
    print("SMA values (first 10):", sma_10[:10])
    
    # List some available TA-Lib functions
    print("\nAvailable Technical Analysis Functions:")
    print("- Moving Averages:", ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA'])
    print("- Momentum Indicators:", ['RSI', 'MACD', 'StochRSI', 'ADX'])
    print("- Volatility Indicators:", ['ATR', 'NATR'])

if __name__ == "__main__":
    verify_talib()