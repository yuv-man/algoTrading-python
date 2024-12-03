import pandas as pd
import numpy as np
from typing import Callable, Dict, Any
from datetime import datetime

class Backtester:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.trades = None
        
    def calculate_metrics(self, trades_df):
        """Calculate metrics for a given set of trades"""
        if len(trades_df) == 0:
            return {
                "End Date": "",
                "Candle Time": "",
                "Short": "",
                "Long": "",
                "Total": 0,
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
            
        profit_trades = trades_df[trades_df['PnL'] > 0]
        loss_trades = trades_df[trades_df['PnL'] < 0]
        even_trades = trades_df[trades_df['PnL'] == 0]
        
        avg_win = profit_trades['PnL'].mean() if len(profit_trades) > 0 else 0
        avg_loss = loss_trades['PnL'].mean() if len(loss_trades) > 0 else 0
        
        # Calculate trends
        trades_df['Trend'] = trades_df['Position'].diff().ne(0).cumsum()
        trades_df['Date'] = trades_df.index.date
        intraday_trends = trades_df.groupby('Date')['Trend'].nunique().sum()
        
        return {
            "End Date": trades_df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            "Candle Time": f"{(self.data.index[1] - self.data.index[0]).seconds // 60}",
            "Short": self.short_window,
            "Long": self.long_window,
            "Total": len(trades_df),
            "Net Profit": round(trades_df['PnL'].sum(), 2),
            "Gross Profit": round(profit_trades['PnL'].sum() if len(profit_trades) > 0 else 0, 2),
            "Gross Loss": round(loss_trades['PnL'].sum() if len(loss_trades) > 0 else 0, 2),
            "Max Loss": round(trades_df['PnL'].min() if len(trades_df) > 0 else 0, 2),
            "% Profit": round(len(profit_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0, 2),
            "Number of Trades": len(trades_df),
            "Number of Profit Trades": len(profit_trades),
            "Number of Loss Trades": len(loss_trades),
            "Number of Even Trades": len(even_trades),
            "Number of Trends": len(trades_df['Trend'].unique()),
            "Number of Trends Intra Day": intraday_trends,
            "Avg Trade": round(trades_df['PnL'].mean() if len(trades_df) > 0 else 0, 2),
            "Avg Winning Trade": round(avg_win, 2),
            "Avg Losing Trade": round(avg_loss, 2),
            "Ratio Avg Win/Avg Loss": round(abs(avg_win / avg_loss) if avg_loss != 0 else 0, 2)
        }

    def backtest(self, strategy: Callable, **Args) -> Dict[str, Any]:
        """Runs backtest and returns metrics for total, long, and short trades"""
        self.short_window = short_window
        self.long_window = long_window
        
        # Generate signals
        self.data['Signal'] = strategy(self.data, **Args)
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

    def print_results(self, results: Dict[str, Dict[str, Any]]):
        """Print results in a three-column format"""
        metrics_order = [
            "End Date", "Candle Time", "Short", "Long", "Total",
            "Net Profit", "Gross Profit", "Gross Loss", "Max Loss", "% Profit",
            "Number of Trades", "Number of Profit Trades", "Number of Loss Trades",
            "Number of Even Trades", "Number of Trends", "Number of Trends Intra Day",
            "Avg Trade", "Avg Winning Trade", "Avg Losing Trade", "Ratio Avg Win/Avg Loss"
        ]
        
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
            
            # Format values based on type
            if isinstance(total_val, float):
                print(f"{metric:<25} {total_val:>25.2f} {long_val:>25.2f} {short_val:>25.2f}")
            else:
                print(f"{metric:<25} {str(total_val):>25} {str(long_val):>25} {str(short_val):>25}")
