import yfinance as yf
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates
from datetime import datetime

class StockTrendAnalyzer:
    def __init__(self, symbol, start_date, end_date, stock_data, interval='1d', period=None):
        """Initialize as before"""
        self.symbol = symbol
        self.interval = interval
        self.period = period
        self.data = stock_data 
        self.peaks = []
        self.troughs = []
        self.uptrends = []
        self.downtrends = []
        self.data['CandleType'] = ''
        self.data['ReversalType'] = ''
    
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

        # Bullish candle
        if (current['Close'] > current['Open'] and
            current['High'] > previous['High'] and
            current['Low'] > previous['Low']):
            self.data.loc[current_idx, 'CandleType'] = 'GreenBar'

        # Bearish candle
        if (current['Close'] < current['Open'] and
            current['High'] < previous['High'] and
            current['Low'] < previous['Low']):
            self.data.loc[current_idx, 'CandleType'] = 'RedBar'

        # Green doji star
        if (in_middle_third(current) and
            current['High'] > previous['High'] and
            current['Low'] > previous['Low']):
            self.data.loc[current_idx, 'ReversalType'] = 'GreenDoji'

        # Red doji star
        if (in_middle_third(current) and
            current['High'] < previous['High'] and
            current['Low'] < previous['Low']):
            self.data.loc[current_idx, 'ReversalType'] = 'RedDoji'

        # Green hammer candle
        if (in_higher_third(current) and
            current['High'] > previous['Low']):
            self.data.loc[current_idx, 'ReversalType'] = 'GreenHammer'

        # Red hammer candle
        if (in_higher_third(current) and
            current['High'] < previous['Low']):
            self.data.loc[current_idx, 'ReversalType'] = 'RedHammer'

        # Inverted hammer candle
        if (in_lower_third(current) and
            current['Low'] > previous['High']):
            self.data.loc[current_idx, 'ReversalType'] = 'InvertedHammerCandle'

        # Green key reversal
        if (current['High'] < previous['High'] and
            current['Low'] < previous['Low'] and
            current['Open'] < previous['Close'] and
            current['Close'] > previous['Close']):
            self.data.loc[current_idx, 'ReversalType'] = 'GreenKR'

        # Red key reversal
        if (current['High'] > previous['High'] and
            current['Low'] > previous['Low'] and
            current['Open'] > previous['Close'] and
            current['Close'] < previous['Close']):
            self.data.loc[current_idx, 'ReversalType'] = 'RedKR'

        # Green outside key reversal
        if (current['High'] > previous['High'] and
            current['Low'] < previous['Low'] and
            current['Open'] > current['Close']):
            self.data.loc[current_idx, 'ReversalType'] = 'GreenOKR'

        # Red outside key reversal
        if (current['High'] > previous['High'] and
            current['Low'] < previous['Low'] and
            current['Open'] < current['Close']):
            self.data.loc[current_idx, 'ReversalType'] = 'RedOKR'

    def find_peaks_and_troughs(self):
        """
        Identifies peaks and troughs based on candlestick patterns and price movement.
        Enforces strict alternation between peaks and troughs.
        Always takes the higher peak or the lower trough if consecutive points are found.
        """
        self.peaks = []
        self.troughs = []
        last_point_type = None  # Tracks if the last point was 'peak', 'trough', or None
    
        # Handle first candle
        first_idx = self.data.index[0]
        current = self.data.loc[first_idx]
        self.data.loc[first_idx, 'Move'] = 'up' if current['Open'] < current['Close'] else 'down'
    
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
                continue  # Skip further processing for inside bars
    
            # Check for outside bar
            is_outside_bar = current['High'] > previous['High'] and current['Low'] < previous['Low']
            if is_outside_bar:
                if current['Open'] > current['Close']:
                    self.data.loc[current_idx, 'ReversalType'] = 'RedOKR'
                else:
                    self.data.loc[current_idx, 'ReversalType'] = 'GreenOKR'
    
            if i > 1:  # Need at least 3 bars for peak/trough detection
                prev_prev_idx = self.data.index[i - 2]
    
                # Skip if previous bar was an inside bar
                if self.data.loc[previous_idx, 'ReversalType'] == 'insidebar':
                    continue
    
                # Peak Detection
                if (self.data.loc[previous_idx, 'CandleMove'] == 'down' or 
                    self.data.loc[previous_idx, 'ReversalType'] in ['Doji', 'InverterHammer', 'RedKR']):
    
                    if last_point_type == 'trough':  # Alternation rule
                        self.peaks.append(i - 2)  # Previous of previous bar is peak
                        last_point_type = 'peak'
                    elif last_point_type == 'peak':  # Two consecutive peaks
                        # Replace last peak if the current one is higher
                        last_peak_idx = self.peaks[-1]
                        if self.data.loc[i - 2, 'High'] > self.data.loc[last_peak_idx, 'High']:
                            self.peaks[-1] = i - 2
    
                # Trough Detection
                if (self.data.loc[previous_idx, 'CandleMove'] == 'up' or 
                    self.data.loc[previous_idx, 'ReversalType'] in ['Doji', 'Hammer', 'GreenKR']):
    
                    if last_point_type == 'peak':  # Alternation rule
                        self.troughs.append(i - 2)  # Previous of previous bar is trough
                        last_point_type = 'trough'
                    elif last_point_type == 'trough':  # Two consecutive troughs
                        # Replace last trough if the current one is lower
                        last_trough_idx = self.troughs[-1]
                        if self.data.loc[i - 2, 'Low'] < self.data.loc[last_trough_idx, 'Low']:
                            self.troughs[-1] = i - 2
    
                # Handle outside bars as peaks or troughs
                if is_outside_bar:
                    if current['Open'] > current['Close']:  # Peak
                        if last_point_type == 'trough':
                            self.peaks.append(i)  # Current bar is peak
                            last_point_type = 'peak'
                        elif last_point_type == 'peak':  # Two consecutive peaks
                            last_peak_idx = self.peaks[-1]
                            if self.data.loc[i, 'High'] > self.data.loc[last_peak_idx, 'High']:
                                self.peaks[-1] = i
                    elif current['Open'] < current['Close']:  # Trough
                        if last_point_type == 'peak':
                            self.troughs.append(i)  # Current bar is trough
                            last_point_type = 'trough'
                        elif last_point_type == 'trough':  # Two consecutive troughs
                            last_trough_idx = self.troughs[-1]
                            if self.data.loc[i, 'Low'] < self.data.loc[last_trough_idx, 'Low']:
                                self.troughs[-1] = i
    
        return self.peaks, self.troughs



    
    def identify_trends(self):
        """Identify uptrends and downtrends based on peak and trough sequences."""
        if self.peaks is None or self.troughs is None:
            self.find_peaks_and_troughs()
            
        closes = self.data['Close'].values
        
        # Combine peaks and troughs with their types for sequential analysis
        all_points = [(idx, closes[idx], 'peak') for idx in self.peaks] + \
                    [(idx, closes[idx], 'trough') for idx in self.troughs]
        all_points.sort(key=lambda x: x[0])
        
        # Reset trends
        self.uptrends = []
        self.downtrends = []
        
        # Analyze sequences for trends
        i = 0
        while i < len(all_points) - 3:
            # Check for uptrend pattern (trough → peak → trough → peak)
            if (i + 3 < len(all_points) and
                all_points[i][2] == 'trough' and
                all_points[i+1][2] == 'peak' and
                all_points[i+2][2] == 'trough' and
                all_points[i+3][2] == 'peak' and
                all_points[i+1][1] < all_points[i+3][1] and  # First peak lower than second peak
                all_points[i][1] < all_points[i+2][1]):      # First trough lower than second trough
                
                start_idx = all_points[i][0]
                
                # Find end of uptrend
                j = i + 5
                while j < len(all_points):
                    if (all_points[j][2] == 'peak' and 
                        all_points[j][1] < all_points[j-2][1]):
                        break
                    j += 2  # Move by 2 to maintain alternation
                
                end_idx = all_points[j-2][0] if j < len(all_points) else all_points[-1][0]
                self.uptrends.append((start_idx, end_idx))
                i = j - 1
                continue
                
            # Check for downtrend pattern (peak → trough → peak → trough)
            if (i + 3 < len(all_points) and
                all_points[i][2] == 'peak' and
                all_points[i+1][2] == 'trough' and
                all_points[i+2][2] == 'peak' and
                all_points[i+3][2] == 'trough' and
                all_points[i+1][1] > all_points[i+3][1] and  # First trough higher than second trough
                all_points[i][1] > all_points[i+2][1]):      # First peak higher than second peak
                
                start_idx = all_points[i][0]
                
                # Find end of downtrend
                j = i + 5
                while j < len(all_points):
                    if (all_points[j][2] == 'trough' and 
                        all_points[j][1] > all_points[j-2][1]):
                        break
                    j += 2  # Move by 2 to maintain alternation
                
                end_idx = all_points[j-2][0] if j < len(all_points) else all_points[-1][0]
                self.downtrends.append((start_idx, end_idx))
                i = j - 1
                continue
            
            i += 1

    def visualize_trends(self, start_date=None, end_date=None):
        """Create a visualization of the stock price with candlesticks and trends."""
        fig, ax = plt.subplots(figsize=(15, 8))
    
        # Convert start_date and end_date to the same timezone as the data, if provided
        if start_date:
            start_date = pd.to_datetime(start_date).tz_localize(self.data['Date'].dt.tz)
        if end_date:
            end_date = pd.to_datetime(end_date).tz_localize(self.data['Date'].dt.tz)
    
        # Filter data by date range if specified
        if start_date and end_date:
            mask = (self.data['Date'] >= start_date) & (self.data['Date'] <= end_date)
            filtered_data = self.data.loc[mask]
        else:
            filtered_data = self.data
    
        # Prepare data for candlestick chart
        
        df_ohlc = filtered_data.reset_index()
        df_ohlc['Date'] = df_ohlc['Date'].map(mpdates.date2num)
        ohlc_data = df_ohlc[['Date', 'Open', 'High', 'Low', 'Close']].values
    
        # Adjust candlestick width based on the interval
        if self.interval == '5m':
            width = 0.0008  # Narrower candles for 5-minute data
        elif self.interval == '1d':
            width = 0.6  # Wider candles for daily data
        else:
            width = 0.02  # General width for other intraday intervals
    
        # Plot candlesticks
        candlestick_ohlc(ax, ohlc_data, width=width, 
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
    
        # Highlight uptrends and downtrends on the plot
        for start_idx, end_idx in self.uptrends:
            if self.data.index[start_idx] in filtered_data.index and self.data.index[end_idx] in filtered_data.index:
                start_date = df_ohlc.loc[start_idx, 'Date']
                end_date = df_ohlc.loc[end_idx, 'Date']
                ax.axvspan(start_date, end_date, alpha=0.2, color='green', label='Uptrend')
    
        for start_idx, end_idx in self.downtrends:
            print(start_idx)
            if self.data.index[start_idx] in filtered_data.index and self.data.index[end_idx] in filtered_data.index:
                start_date = df_ohlc.loc[start_idx, 'Date']
                end_date = df_ohlc.loc[end_idx, 'Date']
                ax.axvspan(start_date, end_date, alpha=0.2, color='red', label='Downtrend')
    
        # Customize the x-axis for intraday intervals
        if self.interval == '5m':
            ax.xaxis.set_major_formatter(mpdates.DateFormatter('%H:%M'))  # Show only time for 5m candles
            ax.xaxis.set_major_locator(mpdates.MinuteLocator(interval=30))  # Major ticks every 30 minutes
            ax.xaxis.set_minor_locator(mpdates.MinuteLocator(interval=5))  # Minor ticks every 5 minutes
        elif self.interval != '1d':
            ax.xaxis.set_major_formatter(mpdates.DateFormatter('%Y-%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mpdates.HourLocator(interval=1))  # Major ticks every hour
            ax.xaxis.set_minor_locator(mpdates.MinuteLocator(interval=15))  # Minor ticks every 15 minutes
        else:
            ax.xaxis.set_major_formatter(mpdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mpdates.AutoDateLocator())
    
        # Customize the plot
        plt.title(f'{self.symbol} Stock Price Trends ({self.interval} Interval)')
        plt.xlabel('Date')
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

    def plotly(self):
        self.data['Bullish'] = self.data['Close'] >= self.data['Open']
    
        # Define colors
        increasing_color = '#26A69A'  # Green for bullish
        decreasing_color = '#EF5350'  # Red for bearish
        
        # Create figure
        fig = go.Figure()
        
        # Add candlesticks
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                increasing_line_color=increasing_color,
                decreasing_line_color=decreasing_color,
                name='Price'
            )
        )
        
        # Add volume bars
        colors = [increasing_color if bullish else decreasing_color for bullish in self.data['Bullish']]
        fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.3,
                yaxis='y2'
            )
        )
        
        # Detect peaks and troughs
        peaks = self.peaks
        troughs = self.troughs
        
        # Add peaks markers
        fig.add_trace(go.Scatter(
            x=self.data.index[peaks],
            y=self.data['Close'].iloc[peaks],
            mode='markers',
            name='Peaks',
            marker=dict(
                symbol='triangle-down',
                size=10,
                color='rgba(255, 182, 193, 0.9)',
                line=dict(color='red', width=2)
            )
        ))
        
        # Add troughs markers
        fig.add_trace(go.Scatter(
            x=self.data.index[troughs],
            y=self.data['Close'].iloc[troughs],
            mode='markers',
            name='Troughs',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color='rgba(182, 255, 193, 0.9)',
                line=dict(color='green', width=2)
            )
        ))
        fig.show()

    def visualize_trends_plotly(self, start_date=None, end_date=None):
        """Create an interactive visualization of the stock price with candlesticks, peaks, and trends using Plotly."""
        if 'Date' not in self.data.columns:
            self.data = self.data.reset_index()  # Reset index if 'Date' is part of the index
            self.data.rename(columns={self.data.index.name: 'Date'}, inplace=True)

        
        # Convert start_date and end_date to the same timezone as the data, if provided
        if start_date:
            start_date = pd.to_datetime(start_date).tz_localize(self.data['Date'].dt.tz)
        if end_date:
            end_date = pd.to_datetime(end_date).tz_localize(self.data['Date'].dt.tz)
        
        # Filter data by date range if specified
        if start_date and end_date:
            mask = (self.data['Date'] >= start_date) & (self.data['Date'] <= end_date)
            filtered_data = self.data.loc[mask]
        else:
            filtered_data = self.data
    
        # Create a Plotly figure
        fig = go.Figure()
    
        # Add candlestick data
        fig.add_trace(go.Candlestick(
            x=filtered_data['Date'],
            open=filtered_data['Open'],
            high=filtered_data['High'],
            low=filtered_data['Low'],
            close=filtered_data['Close'],
            name='Candlesticks'
        ))
    
        # Add peaks (if available)
        if self.peaks:
            fig.add_trace(go.Scatter(
                x=filtered_data.iloc[self.peaks]['Date'],
                y=filtered_data.iloc[self.peaks]['High'],
                mode='markers',
                marker=dict(size=10, color='green', symbol='triangle-up'),
                name='Peaks'
            ))
    
        # Add troughs (if available)
        if self.troughs:
            fig.add_trace(go.Scatter(
                x=filtered_data.iloc[self.troughs]['Date'],
                y=filtered_data.iloc[self.troughs]['Low'],
                mode='markers',
                marker=dict(size=10, color='red', symbol='triangle-down'),
                name='Troughs'
            ))
    
        # Highlight uptrends
        for start_idx, end_idx in self.uptrends:
            if start_idx in filtered_data.index and end_idx in filtered_data.index:
                fig.add_vrect(
                    x0=filtered_data['Date'].iloc[start_idx],
                    x1=filtered_data['Date'].iloc[end_idx],
                    fillcolor="green",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    annotation_text="Uptrend",
                    annotation_position="top left"
                )
    
        # Highlight downtrends
        for start_idx, end_idx in self.downtrends:
            if start_idx in filtered_data.index and end_idx in filtered_data.index:
                fig.add_vrect(
                    x0=filtered_data['Date'].iloc[start_idx],
                    x1=filtered_data['Date'].iloc[end_idx],
                    fillcolor="red",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    annotation_text="Downtrend",
                    annotation_position="top left"
                )
    
        # Customize layout
        fig.update_layout(
            title=f'{self.symbol} Stock Price Trends ({self.interval} Interval)',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    
        # Customize x-axis for intraday intervals
        if self.interval == '5m':
            fig.update_xaxes(
                dtick="600000",  # 10-minute intervals
                tickformat="%H:%M:%S",
                showgrid=True
            )
        elif self.interval != '1d':
            fig.update_xaxes(
                tickformat="%Y-%m-%d %H:%M",
                showgrid=True
            )
        else:
            fig.update_xaxes(
                tickformat="%Y-%m-%d",
                showgrid=True
            )
    
        return fig

