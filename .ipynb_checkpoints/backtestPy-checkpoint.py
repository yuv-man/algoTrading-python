import pandas as pd
import numpy as np
from typing import Callable, Dict, Any
from datetime import datetime
import matplotlib.dates as mpdates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mplfinance.original_flavor import candlestick_ohlc

class Backtester:
    def __init__(self, symbol, start_date, end_date, interval, data: pd.DataFrame, initial_capital=10000, trade_size=None, strategy=None, **params):
        self.data = data.copy()
        self.trades = None
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trade_size = trade_size or initial_capital
        self.params = params
        self.symbol = symbol
        self.start_date = start_date 
        self.end_date = end_date 
        self.interval = interval
        self.peaks = params.get("peaks", [])
        self.troughs = params.get("troughs", [])
        self.uptrends = params.get("uptrends", [])
        self.downtrends = params.get("downtrends", [])
        self.tradesInfo = []
        self.strategy_name = "trend_strategy"
        
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
        self.current_capital = self.initial_capital
        self.trade_size = self.current_capital

        """Runs backtest and returns metrics"""
        # Generate signals using the default trend strategy
        self.data['Signal'] = self.generate_signals(**kwargs)
   
        self.data['Position'] = self.data['Signal'].fillna(0)
        self.data['Trade'] = self.data['Position'].diff()
        
        # Calculate trade metrics
        self.trades = self.data[self.data['Trade'] != 0].copy()
        self.trades['Entry_Price'] = self.trades['Close']
        self.trades['Exit_Price'] = self.trades['Close'].shift(-1)
        self.trades['PnL'] = (self.trades['Exit_Price'] - self.trades['Entry_Price']) * self.trades['Position'] * self.trade_size
        
        # Separate long and short trades
        long_trades = self.trades[self.trades['Position'] > 0]
        short_trades = self.trades[self.trades['Position'] < 0]

        #different
        trades_df = pd.DataFrame(self.tradesInfo) 
        total_metrics = self.calculate_metrics(trades_df)
        long_metrics = self.calculate_metrics(trades_df[trades_df['type'] == 'long'])
        short_metrics = self.calculate_metrics(trades_df[trades_df['type'] == 'short'])
        
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
            self.strategy_name = strategy.__name__
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
        # Initialize signal columns
        self.data['Buy_Signal'] = 0
        self.data['Sell_Signal'] = 0  # For short positions
        self.data['Stop_Loss'] = None
        self.data['Position'] = 0  # 1 for long, -1 for short, 0 for no position
    
        in_position = False
        position_type = None  # 'long' or 'short'
        entry_price = None
        current_stop_loss = None
        position = None

        self.current_capital = self.initial_capital
        portfolio_value = [self.initial_capital]
        self.trade_size = self.current_capital
    
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
                    exit_price = self.data['Close'].iloc[i]
                    profit = (exit_price - position['entry_price']) / position['entry_price']
                    profit_in_money = profit * self.trade_size
                    self.current_capital += profit_in_money
                    trade = {
                        'type': 'long',
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'profit_percent': profit * 100,
                        'profit_money': profit_in_money,
                        'open_date': position['date'],
                        'close_date': self.data.iloc[i].name
                    }
                    self.tradesInfo.append(trade)
                    continue
    
                elif position_type == 'short' and current_price > current_stop_loss:
                    # Stop-loss hit for short position
                    self.data.loc[current_idx, 'Buy_Signal'] = 1
                    in_position = False
                    position_type = None
                    current_stop_loss = None
                    self.data.loc[current_idx, 'Position'] = 0
                    exit_price = self.data['Close'].iloc[i]
                    profit = (exit_price - position['entry_price']) / position['entry_price'] * -1
                    profit_in_money = profit * self.trade_size
                    self.current_capital += profit_in_money
                    trade = {
                        'type': 'short',
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'entry_idx': position['entry_idx'],
                        'exit_idx': i,
                        'profit_percent': profit * 100,
                        'profit_money': profit_in_money,
                        'open_date': position['date'],
                        'close_date': self.data.iloc[i].name
                    }
                    self.tradesInfo.append(trade)
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
                        entry_price = self.data['Close'].iloc[i]
                        position = {'entry_price': entry_price, 'entry_idx': i, 'date': self.data.iloc[i].name}
    
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
                        entry_price = self.data['Close'].iloc[i]
                        position = {'entry_price': entry_price, 'entry_idx': i, 'date': self.data.iloc[i].name}
    
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

    def get_trades_info(self):
        data = pd.DataFrame(self.tradesInfo)
        return data

    def calculate_buy_and_hold(self) -> Dict[str, Dict[str, Any]]:
        """Calculate buy and hold strategy results for comparison."""
        
        # Get first and last prices
        start_price = self.data['Close'].iloc[0]
        end_price = self.data['Close'].iloc[-1]
        
        # Calculate absolute and percentage profit/loss
        total_profit = (end_price - start_price)
        profit_percentage = (total_profit / start_price) * 100
        profit_money = (total_profit / start_price) * self.initial_capital
        return {"total_profit": profit_money, "profit_percentage": profit_percentage }


    def calculate_metrics(self, trades_df):    #avi does not 
        """Calculate metrics for a given set of trades"""
        if len(trades_df) == 0:
            return {
                "Start Date":"",
                "End Date": "",
                "Candle Time": "",
                "Profits": 0,
                "Losses": 0,
                "Net Profit": 0,
                "% Profit": 0,
                "Winning Trades": 0,
                "Max Loss": 0,
                "Number of Trades": 0,
                "Number of Winning Trades": 0,
                "Number of Losing Trades": 0,
                "Number of Even Trades": 0,
                "Number of Trends": 0,
                "Number of Trends Intra Day": 0,
                "Avg Trade": 0,
                "Avg Winning Trade": 0,
                "Avg Losing Trade": 0,
                "Ratio Avg Win/Avg Loss": 0
            }

        trades = trades_df.copy()
        
        profit_trades = trades[trades['profit_percent'] > 0]
        loss_trades = trades[trades['profit_percent'] < 0]
        even_trades = trades[trades['profit_percent'] == 0]
        
        avg_win = profit_trades['profit_money'].mean() if len(profit_trades) > 0 else 0
        avg_loss = loss_trades['profit_money'].mean() if len(loss_trades) > 0 else 0
        
        
        return {
            "Profits": round(profit_trades['profit_money'].sum() if len(profit_trades) > 0 else 0, 2),
            "Losses": round(loss_trades['profit_money'].sum() if len(loss_trades) > 0 else 0, 2),
            "Net Profit": round(trades['profit_money'].sum(), 2),
            "% Profit": round(trades['profit_percent'].sum(),2),
            "Winning Trades": (len(profit_trades) / len(trades))*100,
            "Max Loss": round(trades['profit_money'].min() if len(trades) > 0 else 0, 2),
            "Number of Trades": len(trades),
            "Number of Winning Trades": len(profit_trades),
            "Number of Losing Trades": len(loss_trades),
            "Number of Even Trades": len(even_trades),
            "Number of Trends": len(self.uptrends) + len(self.downtrends),
            "Number of Trends Intra Day": 0,
            "Avg Trade": round(trades['profit_money'].mean() if len(trades) > 0 else 0, 2),
            "Avg Winning Trade": round(avg_win, 2),
            "Avg Losing Trade": round(avg_loss, 2),
            "Ratio Avg Win/Avg Loss": round(abs(avg_win / avg_loss) if avg_loss != 0 else 0, 2)
        }

    def print_results(self, results: Dict[str, Dict[str, Any]]):
        """Print results in a three-column format with negative numbers in red and brackets,
        adding $ for monetary values and % for percentages"""
        metrics_order = [
            "Profits", "Losses",
            "Net Profit", "% Profit", "Winning Trades", "Max Loss", 
            "Number of Trades", "Number of Winning Trades", "Number of Losing Trades",
            "Number of Even Trades", "Number of Trends", "Number of Trends Intra Day",
            "Avg Trade", "Avg Winning Trade", "Avg Losing Trade", "Ratio Avg Win/Avg Loss"
        ]
        
        # Define which metrics should have which symbols
        monetary_metrics = {
            "Net Profit", "Profits", "Losses", "Max Loss",
            "Avg Trade", "Avg Winning Trade", "Avg Losing Trade"
        }
        percentage_metrics = {"% Profit", "Ratio Avg Win/Avg Loss", "Winning Trades"}
        
        # ANSI escape codes for colors
        RED = '\033[91m'
        RESET = '\033[0m'
        
        def format_value(value, metric, width=25):
            """Format a value with special handling for negative numbers, maintaining alignment"""
            if isinstance(value, (int, float)):
                # Determine symbol
                prefix = "$" if metric in monetary_metrics else ""
                suffix = "%" if metric in percentage_metrics else ""
                
                if value < 0:
                    # Format negative numbers with consistent width
                    num_str = f"{prefix}{(value)}{suffix}"
                    formatted = f"{RED}({num_str}){RESET}"
                    # Pad with spaces to maintain alignment
                    padding = width - len(num_str) - 2  # -2 for brackets
                    return " " * max(0, padding) + formatted
                
                num_str = f"{prefix}{value:}{suffix}"
                padding = width - len(num_str)
                return " " * max(0, padding) + num_str
            return f"{str(value):>{width}}"

        buy_and_hold_result = self.calculate_buy_and_hold()
        
        print("\nBACKTEST RESULTS")
        print("=" * 120)

        print(f"Symbol:                  {self.symbol}\t\t\tBuy & Hold Net Profit:   $ {round(buy_and_hold_result['total_profit'],2)}")
        print(f"Start Date:              {self.start_date}\t\tBuy & Hold Profit:     {round(buy_and_hold_result['profit_percentage'],2)}%")
        print(f"End Date:                {self.end_date}")
        print("-" * 120)
        print(f"Interval:                {self.interval}\t\t\tStrategy Profit:      $ {round(self.current_capital-self.initial_capital,2)}")
        print(f"Start Date Capital:      {self.initial_capital}\t\t\tStrategy Yield:       {round((self.current_capital-self.initial_capital)/self.initial_capital*100,2)} %")
        print(f"End Date Capital:        {self.current_capital}")
        print(f"Strategy Name:           {self.strategy_name}")
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
            total_str = format_value(total_val, metric)
            long_str = format_value(long_val, metric)
            short_str = format_value(short_val, metric)
            
            print(f"{metric:<25}{total_str}{long_str}{short_str}")
            
    def visualize_data(self, hideTrends=None, ):
        """Create a visualization of the stock price with candlesticks, trends, and trades."""
        # Create figure with secondary y-axis for volume
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
        fig.set_tight_layout(False)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
    
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        
        # Prepare data
        df_ohlc = self.data.reset_index()
        df_ohlc['Date'] = df_ohlc['Date'].map(mpdates.date2num)
        ohlc_data = df_ohlc[['Date', 'Open', 'High', 'Low', 'Close']].values
        dates_float = df_ohlc['Date'].values
        
        # Plot candlesticks
        candlestick_ohlc(ax1, ohlc_data, width=0.6,
                        colorup='green', colordown='red', alpha=0.7)
        
        # Plot peaks and troughs
        peak_y_positions = self.data['High'].iloc[self.peaks] + (self.data['Close'].iloc[self.peaks] * 0.005)
        ax1.plot(dates_float[self.peaks], peak_y_positions,
                'gv', label='Peaks', markersize=10)
        
        trough_y_positions = self.data['Low'].iloc[self.troughs] - (self.data['Close'].iloc[self.troughs] * 0.005)
        ax1.plot(dates_float[self.troughs], trough_y_positions,
                'r^', label='Troughs', markersize=10)
        
        # Highlight trends
        for start_idx, end_idx in self.uptrends:
            ax1.axvspan(dates_float[start_idx], dates_float[end_idx],
                       alpha=0.2, color='green')
        
        for start_idx, end_idx in self.downtrends:
            ax1.axvspan(dates_float[start_idx], dates_float[end_idx],
                       alpha=0.2, color='red')
        
        # Plot trade entries and exits
        for trade in self.tradesInfo:
            entry_date = dates_float[trade['entry_idx']]
            exit_date = dates_float[trade['exit_idx']]
            
            if trade['type'] == 'long':
                # Plot long entry (green triangle up)
                ax1.plot(entry_date, trade['entry_price'], 'o', color='darkgreen', 
                        markersize=12, label='Long Entry' if 'Long Entry' not in ax1.get_legend_handles_labels()[1] else "")
                # Plot long exit (green triangle down)
                ax1.plot(exit_date, trade['exit_price'], 'o', color='darkgreen', 
                        markersize=12, label='Long Exit' if 'Long Exit' not in ax1.get_legend_handles_labels()[1] else "")
                # Draw connecting line
                ax1.plot([entry_date, exit_date], [trade['entry_price'], trade['exit_price']], 
                        '--', color='darkgreen', alpha=0.5)
            else:  # short trade
                # Plot short entry (red triangle down)
                ax1.plot(entry_date, trade['entry_price'], 'o', color='darkred', 
                        markersize=12, label='Short Entry' if 'Short Entry' not in ax1.get_legend_handles_labels()[1] else "")
                # Plot short exit (red triangle up)
                ax1.plot(exit_date, trade['exit_price'], 'o', color='darkred', 
                        markersize=12, label='Short Exit' if 'Short Exit' not in ax1.get_legend_handles_labels()[1] else "")
                # Draw connecting line
                ax1.plot([entry_date, exit_date], [trade['entry_price'], trade['exit_price']], 
                        '--', color='darkred', alpha=0.5)
                
            # Add profit/loss annotation
            mid_date = entry_date + (exit_date - entry_date)/2
            y_pos = max(trade['entry_price'], trade['exit_price'])
            profit_text = f"{trade['profit_percent']:.1f}%\n${trade['profit_money']:.1f}"
            ax1.annotate(profit_text, 
                        xy=(mid_date, y_pos),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                fc='yellow' if trade['profit_money'] > 0 else 'red',
                                alpha=0.3),
                        color='green' if trade['profit_money'] > 0 else 'red')
        
        # Plot volume bars
        colors = ['green' if close >= open else 'red' 
                  for close, open in zip(self.data['Close'], self.data['Open'])]
        ax2.bar(dates_float, self.data['Volume'], color=colors, alpha=0.7, width=0.6)
        
        # Customize the plots
        ax1.xaxis.set_major_formatter(mpdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mpdates.AutoDateLocator())
        ax1.set_ylabel('Price')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        return plt




