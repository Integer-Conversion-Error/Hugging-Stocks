# Hugging Stocks: Exploring AI/ML for Stock Market Insights

## Motivation

The stock market is a complex system influenced by a vast array of factors, including economic indicators, company performance, geopolitical events, and investor sentiment. Predicting market movements with consistent accuracy remains a significant challenge. The rise of Artificial Intelligence (AI) and Machine Learning (ML) offers powerful new tools to analyze large datasets, identify subtle patterns, and potentially gain novel insights that traditional methods might miss.

This repository documents a personal learning journey into applying these techniques to stock market data. The goal is not to create a definitive prediction engine, but rather to explore *how* different AI/ML approaches can be used, understand their strengths and weaknesses in this context, and learn about the practicalities of data handling and model implementation in finance.

## General Approach

The project explores several distinct angles of AI/ML application:

1.  **Sentiment Analysis:** Gauging the overall mood or opinion expressed in text data (news, social media) regarding specific stocks or the market overall. The hypothesis is that significant shifts in sentiment might precede or correlate with price movements.
2.  **Time Series Modeling:** Treating stock prices as sequential data and applying models designed to capture temporal dependencies and underlying states or regimes (like bull/bear markets) that might govern price behavior.
3.  **External Data Integration:** Incorporating data beyond pure price action, such as large-scale event databases (GDELT) or expert opinions (analyst ratings), to see if they provide additional predictive power.

The focus is on experimentation and understanding the process, rather than achieving state-of-the-art predictive accuracy.

## Learning Objectives

*   **Diverse Analytical Approaches:** Investigate and implement multiple strategies for processing and interpreting stock-related information, moving beyond traditional technical or fundamental analysis.
*   **Sentiment Analysis Exploration:** Delve into natural language processing (NLP) to gauge market sentiment. This includes using general sentiment analysis models and exploring the feasibility of leveraging large-scale event datasets like GDELT for more nuanced insights.
*   **Time Series Modeling:** Experiment with sequence models, specifically Hidden Markov Models (HMMs), to identify potential patterns or regimes in historical stock price movements.
*   **Analyst Data Integration:** Explore methods to incorporate and analyze data derived from professional stock analysts, such as price targets and ratings, potentially correlating them with market movements.
*   **Practical Implementation & Evaluation:** Gain hands-on experience in coding these models, handling relevant financial data, and critically evaluating their performance and usefulness.

## Code Modules

*   **`stock_sentiment.py`**: Implements a baseline approach to stock sentiment analysis, likely using pre-trained NLP models on news headlines or social media data related to specific stocks.
*   **`stock_sentiment_GDELT.py`**: Focuses specifically on utilizing the Global Database of Events, Language, and Tone (GDELT) project data. This explores whether the broader scope and event-focused nature of GDELT can provide unique sentiment signals relevant to stock performance.
*   **`stock_hmm_analysis.py`**: Contains code for applying Hidden Markov Models to stock price time series. The goal is to uncover underlying market states (e.g., bullish, bearish, volatile) that might not be immediately obvious from price charts alone.
*   **`stock_analyst_pricing.py`**: Dedicated to processing and potentially modeling data related to stock analyst recommendations and price targets. This explores how expert opinions are formed, disseminated, and whether they correlate predictably with future stock performance. It involves analyzing the accuracy of past predictions or identifying consensus trends among analysts.

## Future Exploration (Potential Ideas)

*   Combining signals from different models (e.g., sentiment + HMM state).
*   Exploring more advanced time series models (e.g., LSTMs, Transformers).
*   Incorporating fundamental data (e.g., P/E ratios, earnings reports).
*   Developing visualization tools for model outputs.

**Disclaimer:** The contents of this repository are purely for educational and experimental purposes. They do not constitute financial advice, and any analysis or model output should not be used for making investment decisions. Financial markets are inherently unpredictable, and past performance is not indicative of future results.
