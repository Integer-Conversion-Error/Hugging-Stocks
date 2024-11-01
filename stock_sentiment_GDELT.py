import requests
import yfinance as yf
from transformers import pipeline
import config
import stock_hmm_analysis
from datetime import datetime, timedelta
import calendar
import csv
import os
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar

# Constants
GDELT_BASE_URL = 'https://api.gdeltproject.org/api/v2/doc/doc'
SENTIMENT_MODEL = 'nlptown/bert-base-multilingual-uncased-sentiment'

# Set up sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)

def get_company_name(ticker):
    """Fetch the company name for a given stock ticker using yfinance."""
    
    if ticker == "GOOG":
        return "GOOG"
    elif ticker == "AMZN":
        return "AMZN"

    try:
        company_info = yf.Ticker(ticker).info
        return company_info.get("longName") or ticker  # Fallback to ticker if name not found
    except Exception as e:
        print(f"Error fetching company name for {ticker}: {e}")
        return ticker  # Fallback to ticker symbol if API fails




def clean_query(query):
    """Clean and validate the query for GDELT search criteria."""
    # Remove domain extensions like .com, .net, .org, etc.
    query = re.sub(r"\.com|\.net|\.org|\.io|\.co|\.us|\.uk", "", query, flags=re.IGNORECASE)
    
    # Remove common company suffixes
    company_suffixes = ["inc", "corp", "corporation", "ltd", "llc", "gmbh", "plc", "sa", "limited"]
    query_words = query.split()
    filtered_words = [word for word in query_words if word.lower() not in company_suffixes]

    # Remove any common or overly generic words (if needed, add more words)
    common_words = ["the", "in", "and", "of", "news", "report", "article"]
    filtered_words = [word for word in filtered_words if word.lower() not in common_words]

    # Join the cleaned and validated words back into a single query string
    return " ".join(filtered_words)


def fetch_news(query, from_date, to_date, num_articles=100):
    """Fetch recent news articles related to a company name within a date range using GDELT Document API, excluding specific sources."""
    # Clean the query
    query = clean_query(query)
    
    from_date = from_date.replace("-", "") + "000000"
    to_date = to_date.replace("-", "") + "235959"

    params = {
        'query': query,
        'mode': 'ArtList',
        'startdatetime': from_date,
        'enddatetime': to_date,
        'maxrecords': num_articles,
        'format': 'json',
        'sourcelang': 'English',
    }
    
    response = requests.get(GDELT_BASE_URL, params=params)
    
    if response.status_code != 200:
        print(f"Failed to fetch data for {query} from {from_date} to {to_date}. Status Code: {response.status_code}")
        return []

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        print("Response content is not in JSON format. Here is the response content:")
        print(response.text)
        return []

    articles = data.get('articles', [])
    
    # Filter out articles from "WKRB13 News"
    filtered_articles = []
    for article in articles:
        if article.get('source', '').lower() == 'wkrb13 news':
            continue  # Skip this article
        article['source_strength'] = classify_article(article)  # Classify the article
        filtered_articles.append({
            'title': article.get('title', ''),
            'description': article.get('seendate', ''),
            'url': article.get('url', ''),
            'source_strength': article['source_strength']
        })

    return filtered_articles


def classify_article(article):
    """Classify an article based on its source strength."""
    strong_keywords = [
    "press release", "announced", "executive", "official", "statement", 
    "disclosed", "released", "earnings call", "SEC filing", "government report", 
    "quarterly report", "fiscal report", "financial statement", "official report", 
    "conference call", "CEO", "CFO", "authorized", "confirmed", "verified", 
    "annual report", "shareholder letter", "board of directors", "regulatory filing", 
    "audited", "certified", "corporate filing", "press conference", "official announcement"
]

    moderate_keywords = [
    "analyst", "report", "industry", "expert", "market analysis", 
    "consulting firm", "study", "survey", "projection", "forecast", 
    "research report", "white paper", "summary", "evaluation", 
    "market outlook", "insight", "review", "observation", "interview", 
    "predicted", "estimated", "expected", "analysis", "commentary", 
    "business intelligence", "sector report", "trend analysis", 
    "valuation", "financial analyst", "consultant", "advisory", 
    "assessment", "industry data"
]
    weak_keywords = [
    "rumor", "speculation", "unconfirmed", "alleged", "reported", 
    "insider", "gossip", "leaked", "anonymous source", "unverified", 
    "suspected", "claimed", "suggested", "unsubstantiated", "possible", 
    "hinted", "implied", "likely", "anticipated", "potentially", 
    "unclear", "predicted", "projected", "forecasted", "assumed", 
    "suggestive", "unproven", "indicated", "insinuated", "allegedly", 
    "hypothetical", "possibly", "guesswork", "doubtful", "conjecture"
]

    # Combine title and description, handling missing fields gracefully
    content = f"{article.get('title', '')} {article.get('description', '')}".lower()

    # Check for strong keywords
    if any(keyword in content for keyword in strong_keywords):
        return "Strong"
    # Check for moderate keywords
    elif any(keyword in content for keyword in moderate_keywords):
        return "Moderate"
    # Check for weak keywords
    elif any(keyword in content for keyword in weak_keywords):
        return "Weak"
    # Default classification
    return "Unclassified"


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
    last_day_of_previous_month = today.replace(day=1) - relativedelta(days=1)

    # Initialize the months list
    months = []
    
    # Loop to calculate each month's range
    for _ in range(24):
        # Get the first day of the month for last_day_of_previous_month
        first_day = last_day_of_previous_month.replace(day=1)
        
        # Append (first_day, last_day) to the list, formatted as strings
        months.append((first_day.strftime('%Y-%m-%d'), last_day_of_previous_month.strftime('%Y-%m-%d')))
        
        # Move to the previous month
        last_day_of_previous_month = first_day - relativedelta(days=1)
    
    return months

def save_articles_to_csv(ticker, month, articles):
    """Save articles to a CSV file named based on the ticker and month if the file doesn't already exist."""
    os.makedirs(ticker, exist_ok=True)
    csv_file_path = os.path.join(ticker, f"{ticker}_sentiment_for_{month}.csv")
    
    if os.path.exists(csv_file_path):
        print(f"File {csv_file_path} already exists. Skipping write.")
        return

    headers = ['Title', 'Description', 'URL', 'Source Strength']
    
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for article in articles:
            writer.writerow({
                'Title': article['title'],
                'Description': article['description'],
                'URL': article['url'],
                'Source Strength': article['source_strength']
            })
    print(f"Saved articles to {csv_file_path}")

def main():
    # Collect tickers from user
    pre_tickers = []
    tkr = input("Enter the stock tickers you'd like to see analyzed (-1 to quit): ").upper()  # Convert to uppercase
    while tkr != "-1":
        pre_tickers.append(tkr)
        tkr = input("Enter the stock tickers you'd like to see analyzed (-1 to quit): ").upper()  # Convert to uppercase

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
