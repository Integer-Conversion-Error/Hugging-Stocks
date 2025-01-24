import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
def get_sp500_tickers():
    """
    Scrapes Wikipedia to get the current S&P 500 tickers.
    Returns a list of ticker symbols as strings.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # The main table with id='constituents' contains the ticker data
    table = soup.find('table', id='constituents')
    if not table:
        raise Exception("Could not find the S&P 500 table on the Wikipedia page.")
    
    ticker_symbols = []
    
    # The <tbody> has rows (tr). The first row is the header, so we skip it.
    rows = table.find('tbody').find_all('tr')[1:]
    for row in rows:
        cols = row.find_all('td')
        if len(cols) < 2:
            continue  # Skip any row that doesn't have enough columns
        
        symbol = cols[0].text.strip()
        
        # Yahoo Finance often replaces '.' with '-' in symbols like BRK.B -> BRK-B
        symbol = symbol.replace('.', '-')
        
        ticker_symbols.append(symbol)
    
    return ticker_symbols


def get_tickers_filtered(ticker_list,
                         min_5y_return=None,
                         min_market_cap=None,
                         max_market_cap=None,
                         include_industries=None,
                         include_sectors=None):
    """
    Filters a given list of tickers based on:
      - Minimum 5-year % return
      - Market cap range
      - Desired industries
      - Desired sectors
    
    :param ticker_list: List of ticker symbols to evaluate.
    :param min_5y_return: (float) Minimum percent return over the last 5 years. 
                          e.g., 50 means +50%.
    :param min_market_cap: (float) Minimum market cap (e.g., 1e9 for $1 billion).
    :param max_market_cap: (float) Maximum market cap.
    :param include_industries: (list) List of industry strings you want to include.
    :param include_sectors: (list) List of sector strings you want to include.
    
    :return: DataFrame with the tickers that match the filters, 
             plus relevant info.
    """

    # Container for our results
    filtered_results = []

    for ticker_symbol in ticker_list:
        # Download up to 5 years of data
        data = yf.download(ticker_symbol, period='5y', interval='1d', progress=False)
        
        # If we don't have enough data (or any), skip
        if data.empty or len(data) < 2:
            continue

        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]

        # Calculate 5-year % return
        five_yr_return_pct = ((end_price - start_price) / start_price) * 100
        
        # Apply minimum 5-year return filter (if provided)
        if min_5y_return is not None and five_yr_return_pct < min_5y_return:
            continue
        
        # Now get fundamental info (market cap, industry, sector, etc.)
        ticker_obj = yf.Ticker(ticker_symbol)
        info = ticker_obj.info

        market_cap = info.get("marketCap", None)
        industry = info.get("industry", None)
        sector = info.get("sector", None)

        # Check market cap filters
        if min_market_cap is not None:
            # If market cap is missing or too small, skip
            if market_cap is None or market_cap < min_market_cap:
                continue
        
        if max_market_cap is not None:
            # If market cap is missing or too large, skip
            if market_cap is None or market_cap > max_market_cap:
                continue

        # Check industry filter
        if include_industries is not None:
            if industry not in include_industries:
                continue
        
        # Check sector filter
        if include_sectors is not None:
            if sector not in include_sectors:
                continue
        
        # If we reach here, the stock passes all filters
        filtered_results.append({
            "Ticker": ticker_symbol,
            "5Y Return (%)": five_yr_return_pct,
            "Market Cap": market_cap,
            "Industry": industry,
            "Sector": sector
        })
    
    # Convert to DataFrame for convenience
    df_filtered = pd.DataFrame(filtered_results)
    
    # Sort by 5Y return descending, for instance
    if not df_filtered.empty:
        df_filtered.sort_values(by="5Y Return (%)", ascending=False, inplace=True)
        df_filtered.reset_index(drop=True, inplace=True)

    return df_filtered


def filter_stocks_by_analyst_target(tickers, upside_threshold=20, 
                                    min_market_cap=None, 
                                    max_pe=None):
    """
    Filters a list of tickers based on the user-specified upside threshold (%).
    Optional filters for market cap and P/E ratio can be applied.
    Returns a DataFrame of the stocks that pass the filters.
    """
    results = []
    
    for ticker_symbol in tickers:
        ticker = yf.Ticker(ticker_symbol)
        
        # Attempt to retrieve the relevant info
        info = ticker.info
        print(info)
        break
        current_price = info.get("currentPrice", None)
        target_mean_price = info.get("analyst_price_targets", None)
        market_cap = info.get("marketCap", None)
        pe_ratio = info.get("trailingPE", None)
        
        # If current price or target is missing, skip
        if current_price is None or target_mean_price is None or current_price == 0:
            continue
        
        # Calculate the implied upside
        implied_upside = ((target_mean_price - current_price) / current_price) * 100
        
        # Apply filters
        if implied_upside < upside_threshold:
            continue
        
        # Market cap filter (if provided)
        if min_market_cap is not None and (market_cap is None or market_cap < min_market_cap):
            continue
        
        # P/E ratio filter (if provided)
        if max_pe is not None and (pe_ratio is None or pe_ratio > max_pe):
            continue
        
        results.append({
            "Ticker": ticker_symbol,
            "Current Price": current_price,
            "Target Mean Price": target_mean_price,
            "Implied Upside (%)": implied_upside,
            "Market Cap": market_cap,
            "PE Ratio": pe_ratio
        })
    
    # Convert results to a DataFrame for easy display / sorting
    return pd.DataFrame(results)

def plot_recent_performance(filtered_df, period="6mo"):
    """
    Plots the recent performance for each ticker in the filtered DataFrame.
    period can be something like '1mo', '3mo', '6mo', '1y', '5y', etc.
    """
    for _, row in filtered_df.iterrows():
        ticker_symbol = row["Ticker"]
        data = yf.download(ticker_symbol, period=period, progress=False)
        
        if len(data) == 0:
            print(f"No data to plot for {ticker_symbol}.")
            continue
        
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data["Close"], label=ticker_symbol)
        plt.title(f"{ticker_symbol} Price History - Last {period}")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.show()

def main():
    print("Fetching tickers...")
    sp500_list = get_sp500_tickers()
    
    # Ask user for input thresholds, etc.
    upside_threshold = float(input("Enter the minimum % upside (e.g., 20): ") or 20)
    use_market_cap_filter = input("Filter by minimum market cap? (y/n): ") or "n"
    min_market_cap = None
    if use_market_cap_filter.lower() == "y":
        min_market_cap = float(input("Enter the minimum market cap (e.g., 1e9 for $1B): ") or 1e9)
    
    use_pe_filter = input("Filter by maximum P/E ratio? (y/n): ") or "n"
    max_pe = None
    if use_pe_filter.lower() == "y":
        max_pe = float(input("Enter the maximum P/E ratio (e.g., 40): ") or 40)
    
    # Filter the tickers
    filtered_stocks = filter_stocks_by_analyst_target(
        sp500_list, 
        upside_threshold=upside_threshold, 
        min_market_cap=min_market_cap, 
        max_pe=max_pe
    )
    
    if filtered_stocks.empty:
        print("No stocks matched the criteria.")
        return
    
    print("\nStocks matching the criteria:\n")
    print(filtered_stocks)
    
    plot_choice = input("\nWould you like to see recent performance graphs? (y/n): ") or "n"
    if plot_choice.lower() == "y":
        period_choice = input("Enter period for charts (e.g., '1mo', '3mo', '6mo', '1y'): ") or "6mo"
        plot_recent_performance(filtered_stocks, period=period_choice)

if __name__ == "__main__":
    #tickerlist = get_sp500_tickers()
    #print(tickerlist)
    #get_tickers_filtered(ticker_list=tickerlist)
    main()
