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
import math
import pdfplumber
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from collections import Counter
import numpy as np
import plotly.graph_objects as go
import praw
from Keys import client_id, client_secret, user_agent,username,password
import os

SPY = "SPY.csv"
AAPL_PDF = 'AAPL 10-K'
NVDA_PDF = 'NVDA 10-K.pdf'
PTON_PDF = 'PTON 10-K.pdf'
META_PDF = 'META 10K.pdf'

class Reddit_Analysis:
     def __init__(self):
        self.titles = []
        self.tickers = []
        self.clean_list = []
     def reddit_praw(self):
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent,username=username, password=password)
        subreddit = reddit.subreddit('wallstreetbets')
        top_subreddit = subreddit.new(limit=100)

        for submission in top_subreddit:
            title = submission.title
            title_words = title.split()
            self.titles.append(title_words)
        return print(self.titles)

     def clean_reddit(self):
         large_list = ' '.join(map(str,self.titles))
         clean_list = re.sub(r'[^\w\s]','',large_list)
         self.clean_list = clean_list
         return print(self.clean_list)

     def reddit_symbols(self):
         known_not_stocks = ['UPVOTE','SUPPORT','YOLO','CLASS','ACTION','AGAINST','APES','TENDIES','LOSS','GAIN','WSB',
                             'I','STILL','HEAR','NO','BELL','AGAIN']
         known_stocks = pd.read_csv('stock_tickers.csv')
         tickers = known_stocks['Ticker'].tolist()
         for title in self.clean_list:
             for word in title:
                 if word.isupper() and word not in known_not_stocks:
                     self.tickers.append(word)
                 if word in tickers:
                     self.tickers.append(word)
                     return print(self.tickers)







class Asset:
    '''
    def __init__(self, *components):
        self.components = []
    '''

    def __init__(self, file):
        self.volatility = None
        self.file = file
        self.df = None
        self.var = None
        self.combined_df = pd.DataFrame()

    '''
    def __repr__(self,*args):
        # 0th position is always a str represented by the date
        attributes = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"Asset({attributes})"
    '''

    def create_dataframe(self):
        self.df = pd.read_csv(self.file)
        print(self.df)
        return self.df

    def candlestick(self):
        fig = go.Figure(data=[go.Candlestick(x=self.df['Date'],
                                             open=self.df['Open'], high=self.df['High'],
                                             low=self.df['Low'], close=self.df['Close'])
                              ])
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.show()

    def summarize(self):
        self.var = (self.df.describe())
        # print(self.var)
        return self.var

    def calculate_volatlity(self):
        returns = self.df['Close'].pctchange().dropna()
        self.volatility = np.sqrt(252) * returns.std()
        return self.volatility

    def add_dataframe_column(self, dataframe, column_name):
        """ Trying to add all the close columns into one panda dataframe and return it """
        #self.combined_df =
        column = dataframe[column_name]
        self.combined_df = pd.concat([self.combined_df, column], axis=1)
        return self.combined_df

    def calculate_return(self, port_data, weights):
        for stock in port_data.columns[1:]:
            port_data[stock + '_Return'] = port_data[stock].pct_change()

        port_data['Portfolio_Return'] = port_data.iloc[:, 1:].mul(weights).sum(axis=1)
        
    def standard_deviation(self):
        self.sd = self.var.loc['std']  # standard deviation
        # print(self.sd)
        return self.sd

    def process_file(self):
        self.create_dataframe()
        self.candlestick()
        self.summarize()
        self.sharpe_ratio()


# class Recommendation:




def main():
    # Create the Berkshire Hathaway Portfolio by combining stock data for multiple datasets
    # Import all of the assets in Berkshire Porfolio into one pandas data frame of all of the closing prices
    # portfolio_data = pd.merge(stock1_data, stock2_data, on='Date', how='inner')
    # Merge the stock data into a single DataFrame based on the 'Date' column
    #portfolio_data = pd.merge(stock1_data, stock2_data, on='Date', how='inner')


    '''
    Assets_1 = Asset(SPY)
    Assets_1.process_file()
    summary = Assets_1.summarize()
    print(summary)

    close = Assets_1.close_prices()
    print(close)

    stdev = Assets_1.sharpe_ratio()
    print(stdev)
    '''
    #weights = [0.05] * 20
    #entries = os.listdir('Berkshire')
    #for entry in range(1):
        # ASSE = Asset(entry)

    df = pd.DataFrame()








    # Assets_1.summarize(df)
    # print(Assets_1)
    '''
    analyzer = TextAnalyzer(AAPL_PDF)
    print('AAPL 10-K Analysis')
    analyzer.analyze() # results {'neg': 0.057, 'neu': 0.816, 'pos': 0.127, 'compound': 1.0}
    analyzer = TextAnalyzer(NVDA_PDF)
    print('NVDA 10-K Analysis')
    analyzer.analyze() # results {'neg': 0.042, 'neu': 0.824, 'pos': 0.133, 'compound': 1.0}
    analyzer = TextAnalyzer(PTON_PDF)
    print('Peloton 10-K Analysis')
    analyzer.analyze() # results {'neg': 0.039, 'neu': 0.859, 'pos': 0.102, 'compound': 1.0}
    '''

    #analyzer = TextAnalyzer(META_PDF)
    #print('META 10-K Analysis')
    #scores = analyzer.analyze() # results {'neg': 0.055, 'neu': 0.801, 'pos': 0.144, 'compound': 1.0}
    #analyzer.plot(data=scores)
    reddit_analysis = Reddit_Analysis()
    reddit_analysis.reddit_praw()
    reddit_analysis.reddit_symbols()

if __name__ == '__main__':
    main()
