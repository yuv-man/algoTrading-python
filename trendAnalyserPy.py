import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates
from datetime import datetime

class StockTrendAnalyzer:
    def __init__(self, symbol, start_date, end_date, stock_data, interval=None, period=None):
        """Initialize as before"""
        self.symbol = symbol
        self.data = stock_data
        self.peaks = []
        self.troughs = []
        self.uptrends = []
        self.downtrends = []
        self.data['CandleType'] = ''
        self.data['ReversalType'] = ''
        self.data['CandleMove'] = ''

   
    def enforce_alternation(self, peaks, troughs):
        """
        Ensure strict alternation between peaks and troughs.
        Returns cleaned lists where peaks and troughs properly alternate.
        """
        # Combine peaks and troughs with their types and sort by index
        all_points = [(idx, 'peak') for idx in peaks] + [(idx, 'trough') for idx in troughs]
        all_points.sort(key=lambda x: x[0])
       
        # Initialize cleaned lists
        clean_peaks = []
        clean_troughs = []
       
        # Determine if we should start with peak or trough based on first two points
        if len(all_points) < 2:
            return np.array(clean_peaks), np.array(clean_troughs)
       
        # Find the first valid point
        if all_points[0][1] == all_points[1][1]:  # If first two are the same type
            i = 1
            while i < len(all_points) and all_points[i][1] == all_points[0][1]:
                i += 1
            if i < len(all_points):
                # Take the best point among the sequence of same type
                if all_points[0][1] == 'peak':
                    best_idx = max(range(i), key=lambda x: self.data['Close'].iloc[all_points[x][0]])
                else:
                    best_idx = min(range(i), key=lambda x: self.data['Close'].iloc[all_points[x][0]])
                all_points = [all_points[best_idx]] + all_points[i:]
            else:
                return np.array(clean_peaks), np.array(clean_troughs)
       
        # Process remaining points ensuring alternation
        current_type = all_points[0][1]
        last_added = all_points[0][0]
       
        if current_type == 'peak':
            clean_peaks.append(last_added)
        else:
            clean_troughs.append(last_added)
           
        for idx, point_type in all_points[1:]:
            if point_type != current_type:  # Different type than last point
                if point_type == 'peak':
                    clean_peaks.append(idx)
                else:
                    clean_troughs.append(idx)
                current_type = point_type
                last_added = idx
               
        return np.array(clean_peaks), np.array(clean_troughs)
   
    def identify_candle(self, i):
        """
        Identifies the type of candle for the current row based on the previous row.
       
        Args:
        - i: Index of the current row.
        """
        current_idx = self.data.index[i]      
        previous_idx = self.data.index[i - 1]
        current = self.data.loc[current_idx]
        previous = self.data.loc[previous_idx]
   
        def in_middle_third(candle):
            candle_range = candle['High'] - candle['Low']
            mid_low = candle['Low'] + candle_range / 3
            mid_high = candle['High'] - candle_range / 3
            return mid_low <= candle['Open'] <= mid_high and mid_low <= candle['Close'] <= mid_high
   
        def in_higher_third(candle):
            candle_range = candle['High'] - candle['Low']
            high_threshold = candle['High'] - candle_range / 3
            return candle['Open'] >= high_threshold and candle['Close'] >= high_threshold
   
        def in_lower_third(candle):
            candle_range = candle['High'] - candle['Low']
            low_threshold = candle['Low'] + candle_range / 3
            return candle['Open'] <= low_threshold and candle['Close'] <= low_threshold
       
        # Doji
        if in_middle_third(current):  # Added colon here
            self.data.loc[current_idx, 'ReversalType'] = 'Doji'
           
        # Hammer
        if in_higher_third(current):  # Added colon here
            self.data.loc[current_idx, 'ReversalType'] = 'Hammer'
           
        # Inverted Hammer
        if in_lower_third(current):  # Added colon here
            self.data.loc[current_idx, 'ReversalType'] = 'InverterHammer'
       

       

    def find_peaks_and_troughs(self):
        """
        Identifies peaks and troughs based on candlestick patterns and price movement.
        Strictly enforces alternation - must have a trough between peaks and a peak between troughs.
        """
        self.data = self.data.copy()
        self.data['ReversalType'] = None  # Ensure column exists
        self.data['CandleMove'] = None  # Track movement direction
    
        self.peaks = []
        self.troughs = []
        need_peak = False   # True if the next valid point must be a peak
        need_trough = False # True if the next valid point must be a trough
    
        # Handle the first candle
        first_idx = self.data.index[0]
        current = self.data.loc[first_idx]
        self.data.loc[first_idx, 'CandleMove'] = 'up' if current['Open'] < current['Close'] else 'down'
    
        # Process remaining candles
        for i in range(1, len(self.data)):
            current_idx = self.data.index[i]
            previous_idx = self.data.index[i - 1]
    
            current = self.data.loc[current_idx]
            previous = self.data.loc[previous_idx]
    
            # Determine basic move direction
            if current['High'] > previous['High'] and current['Low'] > previous['Low']:
                self.data.loc[current_idx, 'CandleMove'] = 'up'
            elif current['High'] < previous['High'] and current['Low'] < previous['Low']:
                self.data.loc[current_idx, 'CandleMove'] = 'down'
    
            # Check for inside bar - mark and continue
            if current['High'] <= previous['High'] and current['Low'] >= previous['Low']:
                self.data.loc[current_idx, 'ReversalType'] = 'insidebar'
                continue
    
            # Check for outside bar
            is_outside_bar = current['High'] > previous['High'] and current['Low'] < previous['Low']
            if is_outside_bar:
                if current['Open'] > current['Close']:
                    self.data.loc[current_idx, 'ReversalType'] = 'RedOKR'
                else:
                    self.data.loc[current_idx, 'ReversalType'] = 'GreenOKR'
    
            if i > 1:  # Need at least 3 bars for peak/trough detection
                prev_prev_idx = self.data.index[i - 2]
    
                # Skip if the previous bar was an inside bar
                if self.data.loc[previous_idx, 'ReversalType'] == 'insidebar':
                    continue
    
                # Define peak and trough conditions
                peak_condition = (self.data.loc[previous_idx, 'CandleMove'] == 'down' or
                                  self.data.loc[previous_idx, 'ReversalType'] in ['Doji', 'InverterHammer', 'RedKR'])
                trough_condition = (self.data.loc[previous_idx, 'CandleMove'] == 'up' or
                                    self.data.loc[previous_idx, 'ReversalType'] in ['Doji', 'Hammer', 'GreenKR'])
    
                # Handle peaks
                if (peak_condition or (is_outside_bar and current['Open'] > current['Close'])):
                    if len(self.peaks) == 0 or need_peak:  # Can only add a peak if we need one
                        index_to_add = i - 2 if peak_condition else i
    
                        if not trough_condition:  # Ensure it's not also a trough
                            # Check for higher bars between last trough and this peak
                            if self.troughs:
                                last_trough_idx = self.troughs[-1]
                                highest_idx = max(range(last_trough_idx, index_to_add + 1),
                                                  key=lambda x: self.data.loc[self.data.index[x], 'High'])
                                index_to_add = highest_idx
    
                            self.peaks.append(index_to_add)
                            need_peak = False
                            need_trough = True
    
                # Handle troughs
                if (trough_condition or (is_outside_bar and current['Open'] < current['Close'])):
                    if len(self.troughs) == 0 or need_trough:  # Can only add a trough if we need one
                        index_to_add = i - 2 if trough_condition else i
    
                        if not peak_condition:  # Ensure it's not also a peak
                            # Check for lower bars between last peak and this trough
                            if self.peaks:
                                last_peak_idx = self.peaks[-1]
                                lowest_idx = min(range(last_peak_idx, index_to_add + 1),
                                                 key=lambda x: self.data.loc[self.data.index[x], 'Low'])
                                index_to_add = lowest_idx
    
                            self.troughs.append(index_to_add)
                            need_trough = False
                            need_peak = True
    
        return self.peaks, self.troughs

   
    def identify_trends(self):
        """
        Identifies uptrends and downtrends in stock data based on peaks and troughs.
        Uses self.data DataFrame containing stock data (Open, High, Low, Close, Volume)
        and self.peaks, self.troughs arrays containing indices of peaks and troughs.
       
        Uptrend Pattern: Peak → Trough → Peak → Trough
        - When second trough higher than first trough AND second peak higher than first peak
        - Then trend starts at second trough
        - Ends when current closing price lower than second trough
       
        Downtrend Pattern: Trough → Peak → Trough → Peak
        - When second trough lower than first trough AND second peak lower than first peak
        - Then trend starts at second peak
        - Ends when current closing price higher than second peak
        """
        # Initialize empty lists for trends
        self.uptrends = []    # Will store (start_idx, end_idx) tuples for uptrends
        self.downtrends = []  # Will store (start_idx, end_idx) tuples for downtrends
       
        # Need at least 2 peaks and 2 troughs to identify a trend
        if len(self.peaks) < 2 or len(self.troughs) < 2:
            return
           
        # Initialize variables to track current trend
        current_uptrend_start = None
        current_downtrend_start = None
        last_checked_uptrend_idx = 0    # Initialize to 0 instead of None
        last_checked_downtrend_idx = 0  # Initialize to 0 instead of None
       
        # Iterate through each bar in the data
        for current_idx in range(len(self.data)):
            # Skip if we haven't found a new pattern point since last check
            if (current_idx <= last_checked_uptrend_idx or
                current_idx <= last_checked_downtrend_idx):
                continue
               
            # Uptrend Pattern Detection
            if current_uptrend_start is None:
                # Find a complete Peak → Trough → Peak → Trough sequence before current_idx
                pattern_peaks = [p for p in self.peaks if p < current_idx]
                pattern_troughs = [t for t in self.troughs if t < current_idx]
               
                if len(pattern_peaks) >= 2 and len(pattern_troughs) >= 2:
                    peak1, peak2 = pattern_peaks[-2:]
                    trough1, trough2 = pattern_troughs[-2:]
                   
                    # Check if they form the correct sequence
                    if peak1 < trough1 < peak2 < trough2:
                        # Check start conditions
                        if (self.data['Low'].iloc[trough2] > self.data['Low'].iloc[trough1] and
                            self.data['High'].iloc[peak2] > self.data['High'].iloc[peak1]):
                            # Start trend at second trough
                            current_uptrend_start = trough2
                            last_checked_uptrend_idx = current_idx
           
            # Downtrend Pattern Detection
            if current_downtrend_start is None:
                # Find a complete Trough → Peak → Trough → Peak sequence before current_idx
                pattern_peaks = [p for p in self.peaks if p < current_idx]
                pattern_troughs = [t for t in self.troughs if t < current_idx]
               
                if len(pattern_peaks) >= 2 and len(pattern_troughs) >= 2:
                    trough1, trough2 = pattern_troughs[-2:]
                    peak1, peak2 = pattern_peaks[-2:]
                   
                    # Check if they form the correct sequence
                    if trough1 < peak1 < trough2 < peak2:
                        # Check start conditions
                        if (self.data['Low'].iloc[trough2] < self.data['Low'].iloc[trough1] and
                            self.data['High'].iloc[peak2] < self.data['High'].iloc[peak1]):
                            # Start trend at second peak
                            current_downtrend_start = peak2
                            last_checked_downtrend_idx = current_idx
           
            # Check for trend end conditions
            # Uptrend end check
            if current_uptrend_start is not None:
                pattern_troughs = [t for t in self.troughs if t < current_idx]
                if pattern_troughs:
                    last_pattern_trough = pattern_troughs[-1]
                    if self.data['Close'].iloc[current_idx] < self.data['Low'].iloc[last_pattern_trough]:
                        # End the uptrend at current bar
                        self.uptrends.append((current_uptrend_start, current_idx))
                        current_uptrend_start = None
           
            # Downtrend end check
            if current_downtrend_start is not None:
                pattern_peaks = [p for p in self.peaks if p < current_idx]
                if pattern_peaks:
                    last_pattern_peak = pattern_peaks[-1]
                    if self.data['Close'].iloc[current_idx] > self.data['High'].iloc[last_pattern_peak]:
                        # End the downtrend at current bar
                        self.downtrends.append((current_downtrend_start, current_idx))
                        current_downtrend_start = None
       
        # Handle any open trends at the end of the data
        current_idx = len(self.data) - 1
        if current_uptrend_start is not None:
            self.uptrends.append((current_uptrend_start, current_idx))
        if current_downtrend_start is not None:
            self.downtrends.append((current_downtrend_start, current_idx))
        return self.uptrends, self.downtrends


    # Rest of the class remains the same
    def visualize_trends(self):
        """Create a visualization of the stock price with candlesticks and trends."""
        fig, ax = plt.subplots(figsize=(15, 8))
       
        # Prepare data for candlestick chart
        df_ohlc = self.data.reset_index()
        df_ohlc['Date'] = df_ohlc['Date'].map(mpdates.date2num)
        ohlc_data = df_ohlc[['Date', 'Open', 'High', 'Low', 'Close']].values
       
        # Plot candlesticks
        candlestick_ohlc(ax, ohlc_data, width=0.6,
                        colorup='green', colordown='red', alpha=0.7)
       
        # Plot peaks and troughs
        dates_float = df_ohlc['Date'].values
       
        # For peaks: High + (Close * 0.005)
        peak_y_positions = self.data['High'].iloc[self.peaks] + (self.data['Close'].iloc[self.peaks] * 0.005)
        plt.plot(dates_float[self.peaks], peak_y_positions,
                 'gv', label='Peaks', markersize=10)  # Green upward triangle for peaks
       
        # For troughs: Low - (Close * 0.005)
        trough_y_positions = self.data['Low'].iloc[self.troughs] - (self.data['Close'].iloc[self.troughs] * 0.005)
        plt.plot(dates_float[self.troughs], trough_y_positions,
                 'r^', label='Troughs', markersize=10)  # Red downward triangle for troughs
       
        # Highlight trends
        for start_idx, end_idx in self.uptrends:
            plt.axvspan(dates_float[start_idx], dates_float[end_idx],
                       alpha=0.2, color='green')
           
        for start_idx, end_idx in self.downtrends:
            plt.axvspan(dates_float[start_idx], dates_float[end_idx],
                       alpha=0.2, color='red')
       
        # Customize the plot
        ax.xaxis.set_major_formatter(mpdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mpdates.AutoDateLocator())
        plt.title(f'{self.symbol} Stock Price Trends')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
       
        return plt
        
    def visualize_intraday_trends(self):
        """Create a visualization of the intraday stock price with candlesticks and trends."""
        import matplotlib.pyplot as plt
        from mplfinance.original_flavor import candlestick_ohlc
        import matplotlib.dates as mpdates
    
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Prepare data for candlestick chart
        df_ohlc = self.data.reset_index()
        df_ohlc['Date'] = df_ohlc['Datetime'].map(mpdates.date2num)  # Map intraday datetime to numeric format
        ohlc_data = df_ohlc[['Date', 'Open', 'High', 'Low', 'Close']].values
        
        # Plot candlesticks
        candlestick_ohlc(ax, ohlc_data, width=0.0005,  # Smaller width for high granularity
                         colorup='green', colordown='red', alpha=0.7)
        
        # Plot peaks and troughs
        dates_float = df_ohlc['Date'].values
    
        # For peaks: High 
        offset_factor = 0.001  # Smaller offset closer to the peak/trough
        peak_y_positions = self.data['High'].iloc[self.peaks] + (self.data['Close'].iloc[self.peaks] * offset_factor)
        ax.plot(dates_float[self.peaks], peak_y_positions,
                'gv', label='Peaks', markersize=10)  # Green upward triangle for peaks
        
        trough_y_positions = self.data['Low'].iloc[self.troughs] - (self.data['Close'].iloc[self.troughs] * offset_factor)
        ax.plot(dates_float[self.troughs], trough_y_positions,
                'r^', label='Troughs', markersize=10)
        
        # Highlight trends
        for start_idx, end_idx in self.uptrends:
            ax.axvspan(dates_float[start_idx], dates_float[end_idx],
                       alpha=0.2, color='green', label='Uptrend')
        
        for start_idx, end_idx in self.downtrends:
            ax.axvspan(dates_float[start_idx], dates_float[end_idx],
                       alpha=0.2, color='red', label='Downtrend')
        
        # Customize the plot
        ax.xaxis.set_major_formatter(mpdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mpdates.AutoDateLocator())
        plt.title(f'{self.symbol} Intraday Stock Price Trends')
        plt.xlabel('Date and Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt

   
    def get_trend_summary(self):
        """Generate a summary of identified trends."""
        summary = []
       
        for start_idx, end_idx in self.uptrends:
            start_price = self.data['Close'].iloc[start_idx]
            end_price = self.data['Close'].iloc[end_idx]
            change_pct = ((end_price - start_price) / start_price) * 100
           
            summary.append(
                f"Uptrend: Start: {self.data.index[start_idx].strftime('%Y-%m-%d')} "
                f"(Price: ${start_price:.2f}), End: {self.data.index[end_idx].strftime('%Y-%m-%d')} "
                f"(Price: ${end_price:.2f}), Change: {change_pct:.1f}%"
            )
       
        for start_idx, end_idx in self.downtrends:
            start_price = self.data['Close'].iloc[start_idx]
            end_price = self.data['Close'].iloc[end_idx]
            change_pct = ((end_price - start_price) / start_price) * 100
           
            summary.append(
                f"Downtrend: Start: {self.data.index[start_idx].strftime('%Y-%m-%d')} "
                f"(Price: ${start_price:.2f}), End: {self.data.index[end_idx].strftime('%Y-%m-%d')} "
                f"(Price: ${end_price:.2f}), Change: {change_pct:.1f}%"
            )
           
        return summary
