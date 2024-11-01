import requests
from transformers import pipeline
import pandas as pd
import config
import stock_hmm_analysis
api_keys = config
# Constants
API_KEY = api_keys.NEWS_API_KEY
BASE_URL = 'https://newsapi.org/v2/everything'
SENTIMENT_MODEL = 'nlptown/bert-base-multilingual-uncased-sentiment'

# Set up sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, use_auth_token=config.HUGGING_FACE_KEY)

def fetch_news(ticker, num_articles=10):
    """Fetch recent news articles related to a stock ticker using NewsAPI."""
    params = {
        'q': ticker,
        'apiKey': API_KEY,
        'language': 'en',
        'pageSize': num_articles,
    }
    response = requests.get(BASE_URL, params=params)
    articles = response.json().get('articles', [])
    return [{'title': a['title'], 'description': a['description']} for a in articles]

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

def main():
    # Fetch articles

    pre_tickers = []
    tkr = input("Enter the stock tickers you'd like to see analyzed (-1 to quit): ")
    while ticker != "-1":
        pre_tickers.append(tkr)
        tkr = input("Enter the stock tickers you'd like to see analyzed (-1 to quit): ")


    # List of stocks to analyze
    tickers = stock_hmm_analysis.validate_tickers(pre_tickers)
    for ticker in tickers:
        articles = fetch_news(ticker)
        if not articles:
            print("No articles found for the specified ticker.")
            return
        
        # Analyze and print consensus
        consensus, avg_score = analyze_sentiment(articles)
        print(f"Media consensus on {ticker}: {consensus} (Score: {avg_score})")

main()
