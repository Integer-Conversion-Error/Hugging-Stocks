import requests
import yfinance as yf
from transformers import pipeline
import config
import stock_hmm_analysis
from datetime import datetime, timedelta
import calendar
import csv
import os

# Constants
API_KEY = config.NEWS_API_KEY
BASE_URL = 'https://newsapi.org/v2/everything'
SENTIMENT_MODEL = 'nlptown/bert-base-multilingual-uncased-sentiment'

# Set up sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)

def get_company_name(ticker):
    """Fetch the company name for a given stock ticker using yfinance."""
    try:
        company_info = yf.Ticker(ticker).info
        return company_info.get("longName") or ticker  # Fallback to ticker if name not found
    except Exception as e:
        print(f"Error fetching company name for {ticker}: {e}")
        return ticker  # Fallback to ticker symbol if API fails

def fetch_news(query, from_date, to_date, num_articles=100):
    """Fetch recent news articles related to a company name within a date range using NewsAPI."""
    params = {
        'q': query,
        'apiKey': API_KEY,
        'language': 'en',
        'from': from_date,
        'to': to_date,
        'pageSize': num_articles,
    }
    response = requests.get(BASE_URL, params=params)
    articles = response.json().get('articles', [])
    return [{'title': a['title'], 'description': a['description'], 'url': a['url']} for a in articles]

def analyze_sentiment(articles):
    """Analyze sentiment of each article and return average sentiment score."""
    sentiments = []
    for article in articles:
        # Combine title and description for sentiment analysis
        text = f"{article['title']} {article['description']}"
        sentiment = sentiment_analyzer(text)
        sentiments.append(sentiment[0]['label'])
    
    # Convert sentiments to a simple score for averaging (e.g., 1 to 5 for star rating)
    sentiment_scores = {'1 star': -2, '2 stars': -1, '3 stars': 0, '4 stars': 1, '5 stars': 2}
    scores = [sentiment_scores[s] for s in sentiments]
    avg_score = sum(scores) / len(scores) if scores else 0

    # Determine consensus
    if avg_score > 1:
        consensus = "Strongly Positive"
    elif avg_score > 0:
        consensus = "Positive"
    elif avg_score < 0:
        consensus = "Negative"
    else:
        consensus = "Neutral"
    
    return consensus, avg_score

def get_past_six_months():
    """Generate a list of the first and last dates for each of the past six months, starting from the end of the previous month."""
    # Get the last day of the previous month
    today = datetime.today()
    first_day_of_current_month = today.replace(day=1)
    last_day_of_previous_month = first_day_of_current_month - timedelta(days=1)

    # Initialize the months list
    months = []
    
    # Loop to calculate each month's range
    for i in range(6):
        # Set the last day of the month i months ago
        last_day = last_day_of_previous_month - timedelta(days=calendar.monthrange(last_day_of_previous_month.year, last_day_of_previous_month.month)[1] * i)
        # First day is the 1st of that month
        first_day = last_day.replace(day=1)
        
        # Append (first_day, last_day) to the list, formatted as strings
        months.append((first_day.strftime('%Y-%m-%d'), last_day.strftime('%Y-%m-%d')))
        
        # Update last_day_of_previous_month to the end of the previous month
        last_day_of_previous_month = first_day - timedelta(days=1)
    
    return months



def save_articles_to_csv(ticker, month, articles):
    """Save articles to a CSV file named based on the ticker and month if the file doesn't already exist."""
    # Create a directory for the ticker if it doesn't exist
    os.makedirs(ticker, exist_ok=True)
    
    # Define the CSV file path
    csv_file_path = os.path.join(ticker, f"{ticker}_{month}.csv")
    
    # Check if the file already exists
    if os.path.exists(csv_file_path):
        print(f"File {csv_file_path} already exists. Skipping write.")
        return  # Do not write if the file already exists

    # Define the CSV headers
    headers = ['Title', 'Description', 'URL']
    
    # Write articles to the CSV file
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for article in articles:
            writer.writerow({
                'Title': article['title'],
                'Description': article['description'],
                'URL': article['url']
            })
    print(f"Saved articles to {csv_file_path}")

def main():
    # Collect tickers from user
    pre_tickers = []
    tkr = input("Enter the stock tickers you'd like to see analyzed (-1 to quit): ")
    while tkr != "-1":
        pre_tickers.append(tkr)
        tkr = input("Enter the stock tickers you'd like to see analyzed (-1 to quit): ")

    # Validate tickers
    tickers = stock_hmm_analysis.validate_tickers(pre_tickers)
    
    # Get date ranges for the past six months
    date_ranges = get_past_six_months()
    
    # Analyze each ticker
    for ticker in tickers:
        company_name = get_company_name(ticker)
        print(f"\nAnalyzing {company_name} ({ticker})...")
        
        for from_date, to_date in date_ranges:
            articles = fetch_news(company_name, from_date, to_date)
            if not articles:
                print(f"No articles found for {company_name} from {from_date} to {to_date}.")
                continue
            
            # Analyze and print consensus
            consensus, avg_score = analyze_sentiment(articles)
            print(f"From {from_date} to {to_date}: Media consensus on {ticker} ({company_name}): {consensus} (Score: {avg_score})")
            
            # Save articles to CSV
            month = datetime.strptime(from_date, '%Y-%m-%d').strftime('%Y-%m')
            save_articles_to_csv(ticker, month, articles)

# Run the main function
if __name__ == "__main__":
    main()
