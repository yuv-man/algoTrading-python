import pandas as pd
import numpy as np
from typing import Callable, Dict, Any
from datetime import datetime

class Backtester:
    def __init__(self, data: pd.DataFrame, initial_capital=10000 , strategy=None, **params):
        self.data = data.copy()
        self.trades = None
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.params = params
        self.peaks = params.get("peaks", [])
        self.troughs = params.get("troughs", [])
        self.uptrends = params.get("uptrends", [])
        self.downtrends = params.get("downtrends", [])
        
    def backtest(self) -> Dict[str, Any]:
        """Run backtest on the trend strategy"""
        return self.run_backtest()
        
    def generate_signals(self, **kwargs) -> pd.Series:
        """
        Trend following strategy based on peaks, troughs and stop losses
        """
        
        # Execute trading strategy and get position data
        strategy_data = self.execute_strategy()
        
        # Return just the Position series
        return strategy_data['Position'] if 'Position' in strategy_data.columns else pd.Series(0, index=self.data.index)

    def run_backtest(self, **kwargs) -> Dict[str, Any]:

        """Runs backtest and returns metrics"""
        # Generate signals using the default trend strategy
        self.data['Signal'] = self.generate_signals(**kwargs)
   
        self.data['Position'] = self.data['Signal'].fillna(0)
        self.data['Trade'] = self.data['Position'].diff()
        
        # Calculate trade metrics
        self.trades = self.data[self.data['Trade'] != 0].copy()
        self.trades['Entry_Price'] = self.trades['Close']
        self.trades['Exit_Price'] = self.trades['Close'].shift(-1)
        self.trades['PnL'] = (self.trades['Exit_Price'] - self.trades['Entry_Price']) * self.trades['Position']
        
        # Separate long and short trades
        long_trades = self.trades[self.trades['Position'] > 0]
        short_trades = self.trades[self.trades['Position'] < 0]
        
        # Calculate metrics for each category
        total_metrics = self.calculate_metrics(self.trades)
        long_metrics = self.calculate_metrics(long_trades)
        short_metrics = self.calculate_metrics(short_trades)
        
        return {
            'Total': total_metrics,
            'Long': long_metrics,
            'Short': short_metrics
        }

    def execute_strategy(self, strategy: Callable = None) -> pd.DataFrame:
        """
        Executes the given trading strategy. Defaults to the trend-following strategy if none is provided.
        
        Args:
            strategy (Callable): A custom strategy function that accepts the data and returns signals.
            
        Returns:
            pd.DataFrame: A DataFrame with columns for buy signals, sell signals, stop losses, and positions.
        """
        # Use the provided strategy if given, otherwise default to the trend strategy
        if strategy:
            signals = strategy(self.data)
            if not isinstance(signals, pd.DataFrame):
                raise ValueError("The strategy function must return a DataFrame with a 'Position' column.")
            
            # Copy signals into the main data
            self.data['Position'] = signals['Position']
            if 'Stop_Loss' in signals.columns:
                self.data['Stop_Loss'] = signals['Stop_Loss']
            if 'Buy_Signal' in signals.columns:
                self.data['Buy_Signal'] = signals['Buy_Signal']
            if 'Sell_Signal' in signals.columns:
                self.data['Sell_Signal'] = signals['Sell_Signal']
            
            return self.data[['Buy_Signal', 'Sell_Signal', 'Stop_Loss', 'Position']]
        
        # Use the default trend strategy if no custom strategy is provided
        return self.trend_strategy()
        

    def trend_strategy(self) -> pd.DataFrame:
        """
        The default trend-following trading strategy with dynamic stop-loss management.
        
        Returns:
            pd.DataFrame: A DataFrame with columns for buy signals, sell signals, stop losses, and positions.
        """
        print(self.data.head())
        # Initialize signal columns
        self.data['Buy_Signal'] = 0
        self.data['Sell_Signal'] = 0  # For short positions
        self.data['Stop_Loss'] = None
        self.data['Position'] = 0  # 1 for long, -1 for short, 0 for no position
    
        in_position = False
        position_type = None  # 'long' or 'short'
        entry_price = None
        current_stop_loss = None
    
        # Process each bar
        for i in range(1, len(self.data)):
            current_idx = self.data.index[i]
            previous_idx = self.data.index[i - 1]
    
            # Check for stop-loss hit
            if in_position and current_stop_loss is not None:
                current_price = self.data['Close'].iloc[i]
    
                if position_type == 'long' and current_price < current_stop_loss:
                    # Stop-loss hit for long position
                    self.data.loc[current_idx, 'Sell_Signal'] = 1
                    in_position = False
                    position_type = None
                    current_stop_loss = None
                    self.data.loc[current_idx, 'Position'] = 0
                    continue
    
                elif position_type == 'short' and current_price > current_stop_loss:
                    # Stop-loss hit for short position
                    self.data.loc[current_idx, 'Buy_Signal'] = 1
                    in_position = False
                    position_type = None
                    current_stop_loss = None
                    self.data.loc[current_idx, 'Position'] = 0
                    continue
    
            # Check for new uptrend
            for start, end in self.uptrends:
                if i == start and not in_position:
                    # Find the last trough before this point
                    previous_troughs = [t for t in self.troughs if t < i]
                    if previous_troughs:
                        last_trough = previous_troughs[-1]
                        entry_price = self.data['Close'].iloc[i]
                        current_stop_loss = self.data['Low'].iloc[last_trough]
    
                        self.data.loc[current_idx, 'Buy_Signal'] = 1
                        self.data.loc[current_idx, 'Stop_Loss'] = current_stop_loss
                        self.data.loc[current_idx, 'Position'] = 1
                        in_position = True
                        position_type = 'long'
    
            # Check for new downtrend
            for start, end in self.downtrends:
                if i == start and not in_position:
                    # Find the last peak before this point
                    previous_peaks = [p for p in self.peaks if p < i]
                    if previous_peaks:
                        last_peak = previous_peaks[-1]
                        entry_price = self.data['Close'].iloc[i]
                        current_stop_loss = self.data['High'].iloc[last_peak]
    
                        self.data.loc[current_idx, 'Sell_Signal'] = 1
                        self.data.loc[current_idx, 'Stop_Loss'] = current_stop_loss
                        self.data.loc[current_idx, 'Position'] = -1
                        in_position = True
                        position_type = 'short'
    
            # Update stop-loss for existing positions
            if in_position:
                if position_type == 'long':
                    # Check for new trough
                    current_troughs = [t for t in self.troughs if t == i]
                    if current_troughs:
                        new_stop_loss = self.data['Low'].iloc[i]
                        if new_stop_loss > current_stop_loss:  # Only move stop-loss up
                            current_stop_loss = new_stop_loss
                            self.data.loc[current_idx, 'Stop_Loss'] = current_stop_loss
    
                elif position_type == 'short':
                    # Check for new peak
                    current_peaks = [p for p in self.peaks if p == i]
                    if current_peaks:
                        new_stop_loss = self.data['High'].iloc[i]
                        if new_stop_loss < current_stop_loss:  # Only move stop-loss down
                            current_stop_loss = new_stop_loss
                            self.data.loc[current_idx, 'Stop_Loss'] = current_stop_loss
    
            # Carry forward the position and stop-loss
            if i > 0 and self.data.loc[current_idx, 'Position'] == 0:
                self.data.loc[current_idx, 'Position'] = self.data.loc[previous_idx, 'Position']
            if i > 0 and pd.isna(self.data.loc[current_idx, 'Stop_Loss']):
                self.data.loc[current_idx, 'Stop_Loss'] = self.data.loc[previous_idx, 'Stop_Loss']
    
        return self.data[['Buy_Signal', 'Sell_Signal', 'Stop_Loss', 'Position']]


    def calculate_metrics(self, trades_df):    #avi does not 
        """Calculate metrics for a given set of trades"""
        if len(trades_df) == 0:
            return {
                "End Date": "",
                "Candle Time": "",
                "Net Profit": 0,
                "Gross Profit": 0,
                "Gross Loss": 0,
                "Max Loss": 0,
                "% Profit": 0,
                "Number of Trades": 0,
                "Number of Profit Trades": 0,
                "Number of Loss Trades": 0,
                "Number of Even Trades": 0,
                "Number of Trends": 0,
                "Number of Trends Intra Day": 0,
                "Avg Trade": 0,
                "Avg Winning Trade": 0,
                "Avg Losing Trade": 0,
                "Ratio Avg Win/Avg Loss": 0
            }

        trades = trades_df.copy()
    
        profit_trades = trades[trades['PnL'] > 0]
        loss_trades = trades[trades['PnL'] < 0]
        even_trades = trades[trades['PnL'] == 0]
        
        avg_win = profit_trades['PnL'].mean() if len(profit_trades) > 0 else 0
        avg_loss = loss_trades['PnL'].mean() if len(loss_trades) > 0 else 0
        
        # Calculate trends using the copied DataFrame
        trades['Trend'] = trades['Position'].diff().ne(0).cumsum()
        trades.loc[:, 'Date'] = trades.index.date  # Using .loc to avoid the warning
        #intraday_trends = trades.groupby('Date')['Trend'].nunique().sum()
        
        return {
            "Start Date": trades.index[0].strftime('%Y-%m-%d %H:%M:%S'),
            "End Date": trades.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            "Candle Time": '1d',
            "Net Profit": round(trades['PnL'].sum(), 2),
            "Gross Profit": round(profit_trades['PnL'].sum() if len(profit_trades) > 0 else 0, 2),
            "Gross Loss": round(loss_trades['PnL'].sum() if len(loss_trades) > 0 else 0, 2),
            "Max Loss": round(trades['PnL'].min() if len(trades) > 0 else 0, 2),
            "% Profit": round(len(profit_trades) / len(trades) * 100 if len(trades) > 0 else 0, 2),
            "Number of Trades": len(trades),
            "Number of Profit Trades": len(profit_trades),
            "Number of Loss Trades": len(loss_trades),
            "Number of Even Trades": len(even_trades),
            "Number of Trends": len(trades['Trend'].unique()),
            "Number of Trends Intra Day": 0,
            "Avg Trade": round(trades['PnL'].mean() if len(trades) > 0 else 0, 2),
            "Avg Winning Trade": round(avg_win, 2),
            "Avg Losing Trade": round(avg_loss, 2),
            "Ratio Avg Win/Avg Loss": round(abs(avg_win / avg_loss) if avg_loss != 0 else 0, 2)
        }

    def print_results(self, results: Dict[str, Dict[str, Any]]):
        """Print results in a three-column format with negative numbers in red and brackets"""
        metrics_order = [
            "End Date", "Candle Time",
            "Net Profit", "Gross Profit", "Gross Loss", "Max Loss", "% Profit",
            "Number of Trades", "Number of Profit Trades", "Number of Loss Trades",
            "Number of Even Trades", "Number of Trends", "Number of Trends Intra Day",
            "Avg Trade", "Avg Winning Trade", "Avg Losing Trade", "Ratio Avg Win/Avg Loss"
        ]
        
        # ANSI escape codes for colors
        RED = '\033[91m'
        RESET = '\033[0m'
        
        def format_value(value, width=25):
            """Format a value with special handling for negative numbers, maintaining alignment"""
            if isinstance(value, (int, float)):
                if value < 0:
                    # Format negative numbers with consistent width
                    num_str = f"{(value):.2f}"
                    formatted = f"{RED}({num_str}){RESET}"
                    # Pad with spaces to maintain alignment
                    padding = width - len(num_str) - 2  # -2 for brackets
                    return " " * max(0, padding) + formatted
                return f"{value:>{width}.2f}"
            return f"{str(value):>{width}}"
        
        print("\nBACKTEST RESULTS")
        print("=" * 120)
        
        # Print header
        print(f"{'Metric':<25} {'Total':>25} {'Long':>25} {'Short':>25}")
        print("-" * 120)
        
        # Print each metric
        for metric in metrics_order:
            total_val = results['Total'][metric]
            long_val = results['Long'][metric]
            short_val = results['Short'][metric]
            
            # Format each column with proper alignment
            total_str = format_value(total_val)
            long_str = format_value(long_val)
            short_str = format_value(short_val)
            
            print(f"{metric:<25}{total_str}{long_str}{short_str}")

