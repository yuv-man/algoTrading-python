import pandas as pd
import datetime as dt
import yfinance as yf
import matplotlib.dates as mdates

class StockData:
    def __init__(self, symbol, start_date, end_date, interval='1d', specific_given_date=None):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
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

        specific_date=self.specific_given_date or yesterday
        specific_date_start = pd.Timestamp(specific_date + " 00:00:00").tz_localize("America/New_York")
        specific_date_end = specific_date_start + pd.Timedelta(days=1)


        specific_date_data = data[(data.index >= specific_date_start) & (data.index < specific_date_end)]
        return daily_data, specific_date_data
