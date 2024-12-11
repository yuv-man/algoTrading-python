import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.dates as mdates
from typing import Tuple, List

class StockData:
    def __init__(self, symbol, start_date, end_date, interval='1d', specific_given_date=None, period=None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.period = period
        self.stock = None
        self.stock_intraday = None
        self.specific_given_date = specific_given_date

    def load_daily_data(self):
        """
        Load stock data, considering intraday intervals with period and start_date.
        """
         # Intraday intervals
            
            # Filter by start_date
         # Daily or higher intervals
        self.stock = yf.download(self.symbol, start=self.start_date, end=self.end_date, interval=self.interval)
        #self.stock.columns = self.stock.columns.droplevel(1)
        print("Data loaded successfully.")
        return self.stock


    def load_intraday_data(self):
        """
        Load intraday stock data and filter for a specific date range or specific date.
        """
        today = dt.datetime.today()
        start_date_intra = (today - dt.timedelta(days=59)).strftime("%Y-%m-%d")
        yesterday = (today - dt.timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        
        daily_end_date = self.specific_given_date or end_date
        end_daily = dt.datetime.strptime(daily_end_date, '%Y-%m-%d').date()
        start_day_one_year= (end_daily - dt.timedelta(days=365)).strftime("%Y-%m-%d")

        daily_data = yf.download(self.symbol, start=start_day_one_year, end=end_daily, interval='1d')
        #daily_data.columns = daily_data.columns.droplevel(1)

        data = yf.download(self.symbol, start=start_date_intra, end=end_date, interval=self.interval)
        #data.columns = data.columns.droplevel(1)

        # Ensure the index is timezone-aware and normalized
        if data.index.tz is None:
        # Localize to UTC
            data.index = data.index.tz_localize("UTC")
        data.index = data.index.tz_convert("America/New_York")

        if self.period is not None and self.specific_given_date is not None:
            start, end, trading_days = self.get_trading_days(self.specific_given_date, self.period)
            specific_date_start = pd.Timestamp(start + " 00:00:00").tz_localize("America/New_York")
            specific_date_end = pd.Timestamp(end + " 00:00:00").tz_localize("America/New_York")
        else:
            print('hello')
            specific_date=self.specific_given_date or yesterday
            specific_date_start = pd.Timestamp(specific_date + " 00:00:00").tz_localize("America/New_York")
            specific_date_end = specific_date_start + pd.Timedelta(days=1)


        specific_date_data = data[(data.index >= specific_date_start) & (data.index < specific_date_end)]
        return daily_data, specific_date_data


    def get_trading_days(self, start_date_str: str, period: str) -> Tuple[str, str, List[str]]:
        """
        Generate a list of trading days from start date for given period.
        
        Args:
            start_date_str (str): Start date in format 'YYYY-MM-DD'
            period (str): Period in format 'Nd' where N is number of days
            
        Returns:
            tuple: (start_date, end_date, list of trading days) in 'YYYY-MM-DD' format
        """
        # Parse start date and period
        start_date = dt.datetime.strptime(start_date_str, '%Y-%m-%d')
        num_days = int(period[:-1])  # Remove 'd' and convert to int
        num_days+=1
        # Define common US stock market holidays
        holidays = [
            '2024-01-01',  # New Year's Day
            '2024-01-15',  # Martin Luther King Jr. Day
            '2024-02-19',  # Presidents Day
            '2024-03-29',  # Good Friday
            '2024-05-27',  # Memorial Day
            '2024-06-19',  # Juneteenth
            '2024-07-04',  # Independence Day
            '2024-09-02',  # Labor Day
            '2024-11-28',  # Thanksgiving Day
            '2024-12-25',  # Christmas Day
        ]
        
        trading_days = []
        current_date = start_date
        end_date = None
        
        while len(trading_days) < num_days:
            # Format current date as string for comparison
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Check if it's a weekday (Monday = 0, Sunday = 6)
            is_weekend = current_date.weekday() >= 5
            is_holiday = date_str in holidays
            
            if not is_weekend and not is_holiday:
                trading_days.append(date_str)
                if len(trading_days) == num_days:
                    end_date = date_str
                
            current_date += dt.timedelta(days=1)
        
        return start_date_str, end_date, trading_days
    