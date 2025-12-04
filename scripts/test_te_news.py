
import tradingeconomics as te
import pandas as pd

TE_API_KEY = 'E35307B525B94DA:9AD0420D5C6C469'
te.login(TE_API_KEY)

try:
    # Try to fetch news
    news = te.getNews(country='United States', limit=5)
    print("News fetched successfully:")
    print(news)
except Exception as e:
    print(f"Error fetching news: {e}")
