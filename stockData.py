import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.dates as mdates
from typing import Tuple, List
import os

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

    def download_intraday_data(self):
        today = dt.datetime.today()
        start_date_intra = (today - dt.timedelta(days=59)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

        data = yf.download(self.symbol, start=start_date_intra, end=end_date, interval=self.interval)
        #data.columns = data.columns.droplevel(1)
        return data



    def load_intraday_data(self):
        """
        Load intraday stock data and filter for a specific date range or specific date.
        """
        today = dt.datetime.today()
        yesterday = (today - dt.timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
        
        daily_end_date = self.specific_given_date or end_date
        end_daily = dt.datetime.strptime(daily_end_date, '%Y-%m-%d').date()
        start_day_one_year= (end_daily - dt.timedelta(days=365)).strftime("%Y-%m-%d")

        daily_data = yf.download(self.symbol, start=start_day_one_year, end=end_daily, interval='1d')
        #daily_data.columns = daily_data.columns.droplevel(1)

        data = self.download_intraday_data()

        # Ensure the index is timezone-aware and normalized
        if data.index.tz is None:
        # Localize to UTC
            data.index = data.index.tz_localize("America/New_York")
        data.index = data.index.tz_convert("America/New_York")

        if self.period is not None and self.specific_given_date is not None:
            start, end, trading_days = self.get_trading_days(self.specific_given_date, self.period)
            specific_date_start = pd.Timestamp(start + " 00:00:00").tz_localize("America/New_York")
            specific_date_end = pd.Timestamp(end + " 00:00:00").tz_localize("America/New_York")
        else:
            specific_date=self.specific_given_date or yesterday
            specific_date_start = pd.Timestamp(specific_date + " 00:00:00").tz_localize("America/New_York")
            specific_date_end = specific_date_start + pd.Timedelta(days=1)


        specific_date_data = data[(data.index >= specific_date_start) & (data.index <= specific_date_end)]
        return daily_data, specific_date_data


    def get_trading_days(self, end_date_str: str, period: str) -> Tuple[str, str, List[str]]:
        """
        Generate a list of trading days ending on the given end date for the specified period.
        
        Args:
            end_date_str (str): End date in format 'YYYY-MM-DD'
            period (str): Period in format 'Nd' where N is the number of days
            
        Returns:
            tuple: (start_date, end_date, list of trading days) in 'YYYY-MM-DD' format
        """
        # Parse end date and period
        end_date = dt.datetime.strptime(end_date_str, '%Y-%m-%d')
        num_days = int(period[:-1])  # Remove 'd' and convert to int
        num_days = num_days+1  # Include the end date in the count
    
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
        current_date = end_date
        start_date = None
    
        while len(trading_days) <= num_days:
            # Format current date as string for comparison
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Check if it's a weekday (Monday = 0, Sunday = 6) and not a holiday
            is_weekend = current_date.weekday() >= 5
            is_holiday = date_str in holidays
            
            if not is_weekend and not is_holiday:
                trading_days.append(date_str)
                if len(trading_days) == num_days:
                    start_date = date_str
            
            current_date -= dt.timedelta(days=1)
        
        # Reverse the list to have chronological order
        trading_days.reverse()
        trading_days.append(end_date)
        
        return start_date, end_date_str, trading_days

    def download_stock_data_to_file(self,output_dir=None):
        """
        Downloads stock data for a given symbol and date range, then saves it to a file.
        
        Args:
            symbol (str): Stock ticker symbol.
            start_date (str): Start date in 'YYYY-MM-DD' format.
            end_date (str): End date in 'YYYY-MM-DD' format.
            output_dir (str, optional): Directory to save the file. Defaults to "stock_data_<symbol>".
            
        Returns:
            str: Path of the saved file.
        """
        # Set the default output directory if none is provided
        if output_dir is None:
            output_dir = f"stock_data_{self.symbol}"
        
        # Create the output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Fetch stock data using yfinance
        if self.interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']: 
            data = self.download_intraday_data()
        else:
            data = self.load_daily_data()
        
        if data.empty:
            raise ValueError(f"No data found for symbol '{symbol}' in the specified date range.")
        
        # Save the data to a CSV file
        file_path = os.path.join(output_dir, f"{self.symbol}_{self.start_date}_to_{self.end_date}.csv")
        data.to_csv(file_path)
        print(f"Data saved to {file_path}")
        
        return file_path

# Example usage
# download_stock_data_to_file("AAPL", "2023-01-01", "2023-12-01")


    