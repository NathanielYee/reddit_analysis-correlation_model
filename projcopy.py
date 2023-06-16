'''
DS 2500 Stock Market Analysis on Volatility, Price, and Sentiment
In this program we will analyze the net change in the underlying price of a stock based
off of the following criteria
- Volatility
- News Sentiment


CBOE (VIX)
ETF

'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import pdfplumber
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from collections import Counter
import numpy as np
import plotly.graph_objects as go
import praw
from Keys import client_id, client_secret, user_agent, username, password
import os
import yfinance as yf
import scipy.stats as stats

SPY = "SPY.csv"
AAPL_PDF = 'AAPL 10-K'
NVDA_PDF = 'NVDA 10-K.pdf'
PTON_PDF = 'PTON 10-K.pdf'
META_PDF = 'META 10K.pdf'


class Reddit_Analysis:
    def __init__(self):
        self.new_titles = []
        self.top_titles = []
        self.tickers = []
        self.clean_list = []

    def reddit_praw(self):
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent, username=username,
                             password=password)
        subreddit = reddit.subreddit('wallstreetbets')
        new_subreddit = subreddit.new(limit=30)
        top_subreddit = subreddit.top(time_filter="day", limit=5)

        for submission in new_subreddit:
            title = submission.title
            title_words = title.split()
            self.new_titles.append(title_words)

        for submission in top_subreddit:
            title = submission.title
            title_words = title.split()
            self.top_titles.append(title_words)
        return print(self.new_titles)  # , print(self.top_titles)

    def clean_reddit(self):
        '''Asked Chat GPT how to remove emojis '''
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)

        for title in self.new_titles:
            clean_title = emoji_pattern.sub('', ' '.join(title))
            self.clean_list.append(clean_title)

        return self.clean_list

    def reddit_symbols(self):
        known_not_stocks = ['UPVOTE', 'SUPPORT', 'YOLO', 'CLASS', 'ACTION', 'AGAINST', 'APES', 'TENDIES', 'LOSS',
                            'GAIN', 'WSB',
                            'I', 'STILL', 'HEAR', 'NO', 'BELL', 'AGAIN']
        known_stocks = pd.read_csv('stock_tickers.csv')
        tickers = known_stocks['Ticker'].tolist()
        for title in self.clean_list:
            for word in title:
                if word in tickers and word not in known_not_stocks:
                    self.tickers.append(word)
                # if word in tickers:
                # self.tickers.append(word)
        return print(self.tickers)

    def title_analysis(self):
        sent = SentimentIntensityAnalyzer()
        sentiment_score = []
        for title in self.clean_list:
            sentiment_score = sent.polarity_scores(title)
        return print(sentiment_score)


class Yahoo:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = None
        self.volatility = None
        self.df = None
        self.var = None
        self.combined_df = pd.DataFrame()

    def get_historical_data(self):
        self.stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        self.stock_data.reset_index(inplace=True)

    def print_data(self):
        print("\nStock:", self.ticker, self.stock_data)

    def get_data_as_dataframe(self):
        return pd.DataFrame(self.stock_data)

    def candlestick(self):
        fig = go.Figure(data=[go.Candlestick(x=self.df['Date'],
                                             open=self.df['Open'], high=self.df['High'],
                                             low=self.df['Low'], close=self.df['Close'])
                              ])
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.show()

    # def summarize(self):
    #     self.var = (self.df.describe())
    #     # print(self.var)
    #     return self.var
    #
    # def calculate_volatlity(self):
    #     returns = self.df['Close'].pctchange().dropna()
    #     self.volatility = np.sqrt(252) * returns.std()
    #     return self.volatility
    #
    # def standard_deviation(self):
    #     self.sd = self.var.loc['std']  # standard deviation
    #     # print(self.sd)
    #     return self.sd
    #
    # def process_file(self):
    #     self.create_dataframe()
    #     self.candlestick()
    #     self.summarize()

def calculate_return(port_data, weights):
    port_return = pd.DataFrame()
    for stock in port_data.columns[1:]:
        port_return[stock + '_Return'] = port_data[stock].pct_change().fillna(0)

    port_return['Portfolio_Return'] = port_return.iloc[:, 1:].mul(weights).sum(axis=1)
    return port_return

def close_column(dataframe, column_name):
    """ Trying to add all the close columns into one panda dataframe and return it """
    df1_close = dataframe[column_name].copy()
    return df1_close

def create_closing(dataframe):
    # Initialize the first pandas dataframe that takes in all the closing prices of each stock in the portfolio
    closing_df = pd.DataFrame()
    # creating the first column for the pandas to help merge each of the pd df based on the date column
    closing_df['Date'] = dataframe['Date']
    return closing_df



def main():
    # Create the Berkshire Hathaway Portfolio by combining stock data for multiple datasets
    # Import all of the assets in Berkshire Porfolio into one pandas data frame of all of the closing prices
    # portfolio_data = pd.merge(stock1_data, stock2_data, on='Date', how='inner')
    # Merge the stock data into a single DataFrame based on the 'Date' column
    # portfolio_data = pd.merge(stock1_data, stock2_data, on='Date', how='inner')

    tickers = ['TSM', 'V', 'MA', 'PG', 'KO', 'UPS', 'AXP', 'C', 'MMC', 'MCK', 'GM', 'OXY', 'BK', 'HPQ', 'MKL', 'GL',
               'ALLY', 'JEF', 'RH', 'LPX']
    start_date = "2016-01-01"
    end_date = "2022-12-31"

    data_fetcher = Yahoo(tickers[0], start_date, end_date)
    data_fetcher.get_historical_data()
    initial = data_fetcher.get_data_as_dataframe()
    closing_df = create_closing(initial)

    for ticker in tickers:
        data_fetcher = Yahoo(ticker, start_date, end_date)
        data_fetcher.get_historical_data()
        #data_fetcher.print_data()
        data = data_fetcher.get_data_as_dataframe()
        df_close = close_column(data, ["Date", "Close"])
        df_close = df_close.rename(columns={"Close": ticker})
        closing_df = closing_df.merge(df_close, on="Date", how="left")
    print(closing_df)

    # weight of each stock in portfolio
    weights = [1 / 20] * 19
    closing_copy = closing_df.copy()
    port_return = calculate_return(closing_copy, weights)
    print(port_return)

    reddit_analysis = Reddit_Analysis()
    reddit_analysis.reddit_praw()
    reddit_analysis.clean_reddit()
    reddit_analysis.title_analysis()


if __name__ == '__main__':
    main()
