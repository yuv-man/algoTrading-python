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
        self.peaks_intraday = params.get("peaks_intraday", [])
        self.troughs_intraday = params.get("troughs_intraday", [])
        self.uptrends_intraday = params.get("uptrends_intraday", [])
        self.downtrends_intraday = params.get("downtrends_intraday", [])
        self.tradesInfo = []
        self.strategy_name = "trend_strategy"
        self.daily_data = params.get("daily_data", None)
        
    def backtest(self, strategy: Callable = None, **kwargs) -> Dict[str, Any]:
        """Run backtest on the trend strategy"""
        return self.run_backtest(strategy, **kwargs)
        
    def generate_signals(self, strategy: Callable = None, **kwargs) -> pd.Series:
        """
        Trend following strategy based on peaks, troughs and stop losses
        """
        
        # Execute trading strategy and get position data
        if strategy is None:
            strategy_data = self.trend_strategy()
        else:
            strategy_data = self.execute_strategy(strategy, **kwargs)
        # Return just the Position series
        return strategy_data['Position'] if 'Position' in strategy_data.columns else pd.Series(0, index=self.data.index)

    def run_backtest(self, strategy: Callable = None, **kwargs) -> Dict[str, Any]:
        self.current_capital = self.initial_capital
        self.trade_size = self.current_capital

        """Runs backtest and returns metrics"""
        # Generate signals using the default trend strategy
        self.data['Signal'] = self.generate_signals(strategy, **kwargs)
   
        self.data['Position'] = self.data['Signal'].fillna(0)
        self.data['Trade'] = self.data['Position'].diff()

        if len(self.tradesInfo) == 0:
            print("No Trades in between these dates")
            return
        
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

    def execute_strategy(self, strategy: Callable = None, **kwargs) -> pd.DataFrame:
        """
        Executes the given trading strategy with customizable parameters.
        
        Args:
            strategy (Callable): Custom strategy function or None for default trend strategy
            **kwargs: Additional parameters to pass to the strategy function
            
        Returns:
            pd.DataFrame: DataFrame with columns for signals and positions
        """
        new = kwargs.get("something", "default_value")
        stop_loss_pct = kwargs.get("stop_loss", None)
        take_profit_pct = kwargs.get("take_profit", None)
        
        # Initialize tracking variables
        in_position = False
        position_type = None
        entry_price = None
        current_stop_loss = None
        current_take_profit = None
        position = None
        
        # Initialize signal columns
        self.data['Buy_Signal'] = 0
        self.data['Sell_Signal'] = 0
        self.data['Stop_Loss'] = None
        self.data['Take_Profit'] = None
        self.data['Position'] = 0
        self.data['Stop_Loss'] = self.data['Stop_Loss'].astype(float)
        
        # Initialize capital
        self.current_capital = self.initial_capital
        self.trade_size = self.current_capital
        
        # Get strategy signals
        self.strategy_name = strategy._name_
        signals = strategy(self.data, **kwargs)
        
        # Validate signals
        if not isinstance(signals, pd.DataFrame):
            raise ValueError("Strategy must return a DataFrame with required columns")
        
        # Process each bar
        for i in range(1, len(self.data)):
            current_idx = self.data.index[i]
            previous_idx = self.data.index[i-1]
            current_price = self.data['Close'].iloc[i]
            
            # Check for exits if in position
            if in_position:
                # Check stop loss
                if stop_loss_pct and position_type == 'long':
                    current_stop_loss = entry_price * (1 - stop_loss_pct)
                    if current_price < current_stop_loss:
                        self.data.loc[current_idx, 'Sell_Signal'] = 1
                        self.data.loc[current_idx, 'Position'] = 0
                        self._record_trade(position, i, current_price, 'long')
                        in_position = False
                        position_type = None
                        continue
                        
                elif stop_loss_pct and position_type == 'short':
                    current_stop_loss = entry_price * (1 + stop_loss_pct)
                    if current_price > current_stop_loss:
                        self.data.loc[current_idx, 'Buy_Signal'] = 1
                        self.data.loc[current_idx, 'Position'] = 0
                        self._record_trade(position, i, current_price, 'short')
                        in_position = False
                        position_type = None
                        continue
                
                # Check take profit
                if take_profit_pct and position_type == 'long':
                    current_take_profit = entry_price * (1 + take_profit_pct)
                    if current_price > current_take_profit:
                        self.data.loc[current_idx, 'Sell_Signal'] = 1
                        self.data.loc[current_idx, 'Position'] = 0
                        self._record_trade(position, i, current_price, 'long')
                        in_position = False
                        position_type = None
                        continue
                        
                elif take_profit_pct and position_type == 'short':
                    current_take_profit = entry_price * (1 - take_profit_pct)
                    if current_price < current_take_profit:
                        self.data.loc[current_idx, 'Buy_Signal'] = 1
                        self.data.loc[current_idx, 'Position'] = 0
                        self._record_trade(position, i, current_price, 'short')
                        in_position = False
                        position_type = None
                        continue
            
            # Process strategy signals
            signal = signals['Position'].iloc[i]
            
            # Enter new position if signal and not in position
            if not in_position and signal != 0:
                entry_price = current_price
                position_type = 'long' if signal > 0 else 'short'
                position = self._enter_position(entry_price, i)
                in_position = True
                
                if signal > 0:
                    self.data.loc[current_idx, 'Buy_Signal'] = 1
                else:
                    self.data.loc[current_idx, 'Sell_Signal'] = 1
                    
                self.data.loc[current_idx, 'Position'] = signal
                
                # Set stop loss and take profit levels
                if stop_loss_pct:
                    self.data.loc[current_idx, 'Stop_Loss'] = (
                        entry_price * (1 - stop_loss_pct) if signal > 0 
                        else entry_price * (1 + stop_loss_pct)
                    )
                
                if take_profit_pct:
                    self.data.loc[current_idx, 'Take_Profit'] = (
                        entry_price * (1 + take_profit_pct) if signal > 0 
                        else entry_price * (1 - take_profit_pct)
                    )
            
            # Exit existing position if signal changes
            elif in_position and signal == 0:
                self.data.loc[current_idx, 'Position'] = 0
                if position_type == 'long':
                    self.data.loc[current_idx, 'Sell_Signal'] = 1
                else:
                    self.data.loc[current_idx, 'Buy_Signal'] = 1
                
                self._record_trade(position, i, current_price, position_type)
                in_position = False
                position_type = None
            
            # Carry forward position and levels
            elif i > 0:
                self.data.loc[current_idx, 'Position'] = self.data.loc[previous_idx, 'Position']
                self.data.loc[current_idx, 'Stop_Loss'] = self.data.loc[previous_idx, 'Stop_Loss']
                self.data.loc[current_idx, 'Take_Profit'] = self.data.loc[previous_idx, 'Take_Profit']
        
        return self.data[['Buy_Signal', 'Sell_Signal', 'Stop_Loss', 'Take_Profit', 'Position']]
    
    def _record_trade(self, position, exit_idx, exit_price, trade_type):
        """
        Records a completed trade into self.tradesInfo.
    
        Args:
            position (dict): Information about the opened position (entry price, index, date).
            exit_idx (int): The index of the exit point.
            exit_price (float): The exit price.
            trade_type (str): Type of the trade ('long' or 'short').
        """
        profit = ((exit_price - position['entry_price']) / position['entry_price'])
        if trade_type == 'short':
            profit *= -1
        profit_in_money = profit * self.trade_size
        self.current_capital += profit_in_money
    
        trade = {
            'type': trade_type,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_idx': position['entry_idx'],
            'exit_idx': exit_idx,
            'profit_percent': profit * 100,
            'profit_money': profit_in_money,
            'open_date': position['date'],
            'close_date': self.data.iloc[exit_idx].name,
        }
        self.tradesInfo.append(trade)
        
    def _enter_position(self, entry_price, idx):
        return {'entry_price': entry_price, 'entry_idx': idx, 'date': self.data.iloc[idx].name}
        
    def trend_strategy(self) -> pd.DataFrame:
        """
        The default trend-following trading strategy with dynamic stop-loss management.
        Closes any open positions at the end date using the closing price.
        
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
                    self._record_trade(position, i, exit_price, 'long')
                    continue
    
                elif position_type == 'short' and current_price > current_stop_loss:
                    # Stop-loss hit for short position
                    self.data.loc[current_idx, 'Buy_Signal'] = 1
                    in_position = False
                    position_type = None
                    current_stop_loss = None
                    self.data.loc[current_idx, 'Position'] = 0
                    exit_price = self.data['Close'].iloc[i]
                    self._record_trade(position, i, exit_price, 'short')
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
                        position = self._enter_position(entry_price, i)
    
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
                        position = self._enter_position(entry_price, i)    
                        
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
    
        # Close any open position at the end of the period
        last_idx = self.data.index[-1]
        last_price = self.data['Close'].iloc[-1]
        
        if in_position:
            if position_type == 'long':
                self.data.loc[last_idx, 'Sell_Signal'] = 1
                self.data.loc[last_idx, 'Position'] = 0
                self._record_trade(position, len(self.data) - 1, last_price, 'long')
            elif position_type == 'short':
                self.data.loc[last_idx, 'Buy_Signal'] = 1
                self.data.loc[last_idx, 'Position'] = 0
                self._record_trade(position, len(self.data) - 1, last_price, 'short')
    
        return self.data[['Buy_Signal', 'Sell_Signal', 'Stop_Loss', 'Position']]

    def _check_stop_loss(self, current_price, position_type, entry_price, stop_loss_pct):
        if position_type == 'long':
            stop_loss = entry_price * (1 - stop_loss_pct)
            return current_price < stop_loss, stop_loss
        elif position_type == 'short':
            stop_loss = entry_price * (1 + stop_loss_pct)
            return current_price > stop_loss, stop_loss
        return False, None

    def get_trades_info(self):
        data = pd.DataFrame(self.tradesInfo)
        return data

    def get_trades_info_per_day(self):
        """
        Calculate and display daily trading statistics based on actual trade profits.
        Uses the profit data from individual trades rather than day start/end prices.
        """
        data = pd.DataFrame(self.tradesInfo)
        
        # Convert dates to datetime
        data['close_date'] = pd.to_datetime(data['close_date'])
        
        # Group by close date to get daily statistics from actual trades
        daily_stats = data.groupby(data['close_date'].dt.date).agg({
            'profit_money': 'sum',    # Sum of all trade profits for the day
            'profit_percent': 'sum',  # Sum of all trade profit percentages
            'type': 'count'          # Number of trades per day
        }).reset_index()
        
        # Format the columns for display
        formatted_stats = daily_stats.copy()
        formatted_stats['profit_money'] = formatted_stats['profit_money'].apply(lambda x: f"${x:,.2f}")
        formatted_stats['profit_percent'] = formatted_stats['profit_percent'].apply(lambda x: f"{x:,.2f}%")
        
        # Rename columns for better display
        formatted_stats.columns = [
            'Date', 'Daily Profit ($)', 'Daily Profit (%)', 'Number of Trades'
        ]
        
        # Print the formatted table
        print("\nDaily Trading Statistics (Based on Actual Trades)")
        print("=" * 80)
        print(formatted_stats.to_string(index=False))
        print("=" * 80)
        
        # Return the unformatted DataFrame for further calculations if needed
        return daily_stats

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
            "Winning Trades": round(len(profit_trades) / len(trades)*100,2),
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
        print(f"End Date Capital:        {round(self.current_capital,2)}")
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

    def visualize_intraday_data(self, prev_bar, last_peak, last_trough):
        """
        Create a visualization of the intraday stock price with candlesticks and trends.
        
        Args:
            prev_bar (dict): Previous bar data with 'High', 'Low', 'Close', 'Open'
            last_peak (float): Last daily peak price
            last_trough (float): Last daily trough price
            
        Returns:
            matplotlib.figure: The complete figure object
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
        fig.set_tight_layout(False)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        # Prepare OHLC data
        df_ohlc = self.data.reset_index()
        df_ohlc['Date'] = pd.to_datetime(df_ohlc['Datetime']).map(mpdates.date2num)
        ohlc_data = df_ohlc[['Date', 'Open', 'High', 'Low', 'Close']].values
        dates_float = df_ohlc['Date'].values
        
        # Plot candlesticks
        candlestick_ohlc(ax1, ohlc_data, width=0.0005, 
                         colorup='green', colordown='red', alpha=0.7)
        
        # Plot peaks and troughs with improved positioning
        def plot_extrema(ax, indices, prices, is_peak=True):
            if len(indices) > 0:
                marker = 'gv' if is_peak else 'r^'
                label = 'Peaks' if is_peak else 'Troughs'
                offset = prices.iloc[indices] * 0.0005 * (1 if is_peak else -1)
                y_positions = prices.iloc[indices] + offset
                ax.plot(dates_float[indices], y_positions, marker, 
                       label=label, markersize=10)
        
        plot_extrema(ax1, self.peaks_intraday, self.data['High'], True)
        plot_extrema(ax1, self.troughs_intraday, self.data['Low'], False)
        
        # Plot trends with better labeling
        def plot_trends(ax, trends, color, label):
            plotted = False
            for start_idx, end_idx in trends:
                ax.axvspan(dates_float[start_idx], dates_float[end_idx],
                          alpha=0.2, color=color, 
                          label=label if not plotted else "_nolegend_")
                plotted = True
        
        plot_trends(ax1, self.uptrends_intraday, 'green', 'Uptrend')
        plot_trends(ax1, self.downtrends_intraday, 'red', 'Downtrend')
        
        # Calculate and apply price range limits
        price_range = self.data['High'].max() - self.data['Low'].min()
        range_limit = 1.5 * price_range
        
        # Plot previous bar levels
        reference_levels = [
            (prev_bar['High'], 'green', '--', 1, 'Prev Bar High'),
            (prev_bar['Low'], 'red', '--', 1, 'Prev Bar Low'),
            (prev_bar['Close'], 'blue', '-', 1, 'Prev Bar Close'),
            (prev_bar['Open'], 'orange', '-', 1, 'Prev Bar Open'),
            (last_peak, 'darkorange', '-', 2.5, 'Last Daily Peak'),
            (last_trough, 'purple', '-', 2.5, 'Last Daily Trough')
        ]
        
        for price, color, style, width, label in reference_levels:
            if abs(price - self.data['High'].max()) <= range_limit:
                ax1.axhline(price, color=color, linestyle=style, 
                           linewidth=width, label=label)
        
        # Plot trades with improved visualization
        def plot_trade(ax, trade, dates):
            entry_date = dates[trade['entry_idx']]
            exit_date = dates[trade['exit_idx']]
            is_long = trade['type'] == 'long'
            color = 'darkgreen' if is_long else 'darkred'
            entry_label = f"{trade['type'].capitalize()} Entry"
            exit_label = f"{trade['type'].capitalize()} Exit"
            
            # Plot entry/exit points
            for date, price, label in [(entry_date, trade['entry_price'], entry_label),
                                     (exit_date, trade['exit_price'], exit_label)]:
                if label not in ax.get_legend_handles_labels()[1]:
                    ax.plot(date, price, 'o', color=color, markersize=12, label=label)
                else:
                    ax.plot(date, price, 'o', color=color, markersize=12)
            
            # Draw connection line
            ax.plot([entry_date, exit_date], 
                    [trade['entry_price'], trade['exit_price']],
                    '--', color=color, alpha=0.5)
            
            # Add profit/loss annotation
            mid_date = entry_date + (exit_date - entry_date)/2
            y_pos = max(trade['entry_price'], trade['exit_price'])
            profit_text = f"{trade['profit_percent']:.1f}%\n${trade['profit_money']:.1f}"
            profit_positive = trade['profit_money'] > 0
            
            ax.annotate(profit_text,
                       xy=(mid_date, y_pos),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.5',
                               fc='yellow' if profit_positive else 'red',
                               alpha=0.3),
                       color='green' if profit_positive else 'red')
        
        for trade in self.tradesInfo:
            plot_trade(ax1, trade, dates_float)
        
        # Plot volume with improved colors
        volume_colors = ['green' if close >= open else 'red'
                        for close, open in zip(self.data['Close'], self.data['Open'])]
        ax2.bar(dates_float, self.data['Volume'], 
                color=volume_colors, alpha=0.7, width=0.0005)
        
        # Style and format the plot
        ax1.xaxis.set_major_formatter(mpdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax1.xaxis.set_major_locator(mpdates.AutoDateLocator())
        ax1.set_title(f'{self.symbol} Intraday Stock Price Trends')
        ax2.set_xlabel('Date and Time')
        ax1.set_ylabel('Price')
        ax2.set_ylabel('Volume')
        
        # Add grid and adjust layout
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Rotate x-axis labels and adjust layout
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                
        return fig




