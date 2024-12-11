import pandas as pd
import numpy as np

class Strategies:
    """
    A class that contains various trading strategies.
    """
    
    @staticmethod
    def sma_crossover_strategy(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        
        """
        Simple Moving Average (SMA) crossover strategy for long positions only.
        Generates buy signals when short-term SMA crosses above long-term SMA,
        and exit signals when short-term SMA crosses below long-term SMA.

        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            **kwargs: Strategy parameters including:
                - sma_s (int): Short-term SMA period (default: 20)
                - sma_l (int): Long-term SMA period (default: 50)
                - take_profit (float): Take profit percentage (default: None)
                - stop_loss (float): Stop loss percentage (default: None)

        Returns:
            pd.DataFrame: DataFrame with Position column (0 for no position, 1 for long)
        """
        # Get parameters from kwargs with defaults
        sma_s = kwargs.get('sma_s', 20)
        sma_l = kwargs.get('sma_l', 50)

        # Calculate SMAs
        data['SMA_Short'] = data['Close'].rolling(window=sma_s).mean()
        data['SMA_Long'] = data['Close'].rolling(window=sma_l).mean()

        # Initialize position column
        data['Position'] = 0

        # Generate signals
        # Long entry signal: Short SMA crosses above Long SMA
        data.loc[data['SMA_Short'] > data['SMA_Long'], 'Position'] = 1

        # Exit signal: Short SMA crosses below Long SMA
        data.loc[data['SMA_Short'] < data['SMA_Long'], 'Position'] = 0

        # Fill NaN values with 0
        data['Position'] = data['Position'].fillna(0)

        return data[['Position']]

    @staticmethod
    def multi_indicator_strategy(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        
        """
        Strategy using 5 technical indicators with fixed 1% stop loss.
        Exits when any 2 indicators show sell signals or at the end of the day.
        """
        # Get indicator parameters (unchanged)
        ema_fast = kwargs.get('ema_fast', 12)
        ema_slow = kwargs.get('ema_slow', 26)
        rsi_period = kwargs.get('rsi_period', 14)
        rsi_oversold = kwargs.get('rsi_oversold', 30)
        rsi_overbought = kwargs.get('rsi_overbought', 70)
        macd_fast = kwargs.get('macd_fast', 12)
        macd_slow = kwargs.get('macd_slow', 26)
        macd_signal = kwargs.get('macd_signal', 9)
        bb_period = kwargs.get('bb_period', 20)
        bb_std = kwargs.get('bb_std', 2)
        
        # Calculate indicators (unchanged)
        data['EMA_Fast'] = data['Close'].ewm(span=ema_fast, adjust=False).mean()
        data['EMA_Slow'] = data['Close'].ewm(span=ema_slow, adjust=False).mean()
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        exp1 = data['Close'].ewm(span=macd_fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=macd_slow, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal_Line'] = data['MACD'].ewm(span=macd_signal, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['Signal_Line']
        
        data['BB_Middle'] = data['Close'].rolling(window=bb_period).mean()
        bb_std_dev = data['Close'].rolling(window=bb_period).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * bb_std_dev)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * bb_std_dev)
        
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        data['Cumulative_PV'] = (data['Close'] * data['Volume']).cumsum()
        data['VWAP'] = data['Cumulative_PV'] / data['Cumulative_Volume']
        
        # Initialize position tracking
        data['Position'] = 0
        data['Entry_Price'] = 0
        data['Entry_Price'] = data['Entry_Price'].astype('float64')
        data['Stop_Loss'] = 0
        data['Stop_Loss'] = data['Stop_Loss'].astype('float64')
    
        position_open = False
        entry_price = 0
        stop_loss = 0
        
        for i in range(1, len(data)):
            current_close = data['Close'].iloc[i]
            
            # Detect end of day
            current_date = data.index[i].date()
            next_date = data.index[i + 1].date() if i + 1 < len(data) else None
            end_of_day = next_date and current_date != next_date
            
            if position_open:
                # Check stop loss first
                if current_close <= stop_loss:
                    position_open = False
                    data.loc[data.index[i], 'Position'] = 0
                    continue
                
                # Count sell signals from indicators
                sell_signals = 0
                
                # EMA sell signal
                if (data['EMA_Fast'].iloc[i] < data['EMA_Slow'].iloc[i] and 
                    data['EMA_Fast'].iloc[i-1] >= data['EMA_Slow'].iloc[i-1]):
                    sell_signals += 1
                
                # RSI sell signal
                if (data['RSI'].iloc[i] > rsi_overbought and 
                    data['RSI'].iloc[i-1] <= rsi_overbought):
                    sell_signals += 1
                
                # MACD sell signal
                if (data['MACD_Hist'].iloc[i] < 0 and 
                    data['MACD_Hist'].iloc[i-1] >= 0):
                    sell_signals += 1
                
                # Bollinger Bands sell signal
                if (data['Close'].iloc[i-1] >= data['BB_Upper'].iloc[i-1] and 
                    data['Close'].iloc[i] < data['BB_Upper'].iloc[i]):
                    sell_signals += 1
                
                # VWAP sell signal
                if (data['Close'].iloc[i] < data['VWAP'].iloc[i] and 
                    data['Close'].iloc[i-1] >= data['VWAP'].iloc[i-1]):
                    sell_signals += 1
                
                # Exit if 2 or more sell signals or at end of day
                if sell_signals >= 2 or end_of_day:
                    position_open = False
                    data.loc[data.index[i], 'Position'] = 0
                    continue
                
                data.loc[data.index[i], 'Position'] = 1
                data.loc[data.index[i], 'Entry_Price'] = entry_price
                data.loc[data.index[i], 'Stop_Loss'] = stop_loss
            
            else:
                # Check for entry signals
                ema_signal = (data['EMA_Fast'].iloc[i] > data['EMA_Slow'].iloc[i] and 
                             data['EMA_Fast'].iloc[i-1] <= data['EMA_Slow'].iloc[i-1])
                
                rsi_signal = (data['RSI'].iloc[i] > rsi_oversold and 
                             data['RSI'].iloc[i-1] <= rsi_oversold)
                
                macd_signal = (data['MACD_Hist'].iloc[i] > 0 and 
                              data['MACD_Hist'].iloc[i-1] <= 0)
                
                bb_signal = (data['Close'].iloc[i-1] <= data['BB_Lower'].iloc[i-1] and 
                            data['Close'].iloc[i] > data['BB_Lower'].iloc[i])
                
                vwap_signal = (data['Close'].iloc[i] > data['VWAP'].iloc[i] and 
                              data['Close'].iloc[i-1] <= data['VWAP'].iloc[i-1])
                
                if any([ema_signal, rsi_signal, macd_signal, bb_signal, vwap_signal]):
                    position_open = True
                    entry_price = current_close
                    stop_loss = entry_price * 0.99  # Fixed 1% stop loss
                    
                    data.loc[data.index[i], 'Position'] = 1
                    data.loc[data.index[i], 'Entry_Price'] = entry_price
                    data.loc[data.index[i], 'Stop_Loss'] = stop_loss
        
        return data[['Position', 'Entry_Price', 'Stop_Loss', 'EMA_Fast', 'EMA_Slow', 
                    'RSI', 'MACD', 'Signal_Line', 'MACD_Hist', 'BB_Upper', 'BB_Lower', 
                    'BB_Middle', 'VWAP']]

    @staticmethod
    def gaps_strategy(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        High-probability gap trading strategy based on momentum and market conditions.
        
        Parameters:
        - data: DataFrame with OHLCV data
        - kwargs: Additional parameters (unused)
        
        Returns:
        - DataFrame with Position column (1 for long, -1 for short, 0 for no position)
        """
        # Create working copy
        df = data.copy()
        
        # Calculate basic metrics
        df['prev_close'] = df['Close'].shift(1)
        df['prev_high'] = df['High'].shift(1)
        df['prev_low'] = df['Low'].shift(1)
        df['prev_open'] = df['Open'].shift(1)
        
        # Calculate gap metrics
        df['gap_size'] = (df['Open'] - df['prev_close']) / df['prev_close'] * 100
        df['abs_gap_size'] = abs(df['gap_size'])
        
        # Calculate trend indicators
        df['sma5'] = df['Close'].rolling(window=5).mean()
        df['sma20'] = df['Close'].rolling(window=20).mean()
        df['trend_up'] = df['sma5'] > df['sma20']
        
        # Calculate volatility
        df['atr'] = df['High'] - df['Low']
        df['avg_atr'] = df['atr'].rolling(window=10).mean()
        df['volatility'] = df['Close'].rolling(window=10).std() / df['Close'].rolling(10).mean() * 100
        
        # Previous day characteristics
        df['prev_day_range'] = df['prev_high'] - df['prev_low']
        df['prev_day_body'] = abs(df['prev_close'] - df['prev_open'])
        df['prev_day_body_ratio'] = df['prev_day_body'] / df['prev_day_range']
        
        # Volume condition
        df['avg_volume'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['avg_volume']
        
        # Initialize position column
        df['Position'] = 0
        
        # Strategy conditions
        long_condition = (
            # Gap size between 0.3% and 0.7%
            (df['gap_size'] > 0.3) & 
            (df['gap_size'] < 0.7) &
            
            # Uptrend confirmation
            (df['trend_up']) &
            
            # Strong previous day momentum
            (df['prev_day_body_ratio'] > 0.8) &
            
            # Low volatility environment
            (df['volatility'] < 1.0) &
            
            # Volume confirmation
            (df['volume_ratio'] > 1.2) &
            
            # Gap doesn't breach previous day's high
            (df['Open'] < df['prev_high'])
        )
        
        short_condition = (
            # Large gap up in downtrend
            (df['gap_size'] > 0.5) &
            (~df['trend_up']) &
            
            # Weak previous day
            (df['prev_day_body_ratio'] < 0.3) &
            
            # High volatility environment
            (df['volatility'] > 1.0) &
            
            # Volume spike
            (df['volume_ratio'] > 1.5)
        )
        
        # Set positions based on conditions
        df.loc[long_condition, 'Position'] = 1
        df.loc[short_condition, 'Position'] = -1
        
        # Avoid consecutive losses
        df['prev_position'] = df['Position'].shift(1)
        df['prev_success'] = ((df['Position'].shift(1) == 1) & 
                             (df['Close'] > df['Open'].shift(1))) | \
                            ((df['Position'].shift(1) == -1) & 
                             (df['Close'] < df['Open'].shift(1)))
        
        # Count consecutive losses
        df['losing_streak'] = (~df['prev_success']).rolling(window=3, min_periods=1).sum()
        
        # No trading after 2 consecutive losses
        df.loc[df['losing_streak'] >= 2, 'Position'] = 0
        
        # Return only Position column
        return df[['Position']]


Strategies.sma_crossover_strategy._name_ = 'SMA_Crossover'
Strategies.multi_indicator_strategy._name_ = 'Multi_Indicator_Strategy'
Strategies.gaps_strategy._name_ = 'Gaps Strategy'
    



    