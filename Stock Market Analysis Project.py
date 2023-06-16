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
#import pdfplumber
#import nltk
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from collections import Counter
import numpy as np
import plotly.graph_objects as go
import praw
from Keys import client_id, client_secret, user_agent,username,password
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
        # Get top 30 posts and then get sent score and compare to return of the 30 days
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent,username=username, password=password)
        subreddit = reddit.subreddit('wallstreetbets')
        new_subreddit = subreddit.new(limit=30)
        top_subreddit = subreddit.top(time_filter="day",limit=5)

        for submission in new_subreddit:
            title = submission.title
            title_words = title.split()
            self.new_titles.append(title_words)


        for submission in top_subreddit:
            title = submission.title
            title_words = title.split()
            self.top_titles.append(title_words)
        return print(self.new_titles)#, print(self.top_titles)

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
        known_not_stocks = ['UPVOTE','SUPPORT','YOLO','CLASS','ACTION','AGAINST','APES','TENDIES','LOSS','GAIN','WSB',
                             'I','STILL','HEAR','NO','BELL','AGAIN']
        known_stocks = pd.read_csv('stock_tickers.csv')
        tickers = known_stocks['Ticker'].tolist()
        for title in self.clean_list:
            for word in title:
                if word in tickers and word not in known_not_stocks:
                    self.tickers.append(word)
                #if word in tickers:
                    #self.tickers.append(word)
        return print(self.tickers)
    def title_analysis(self):
        sent = SentimentIntensityAnalyzer()
        sentiment_score = []
        for title in self.clean_list:
            sentiment_score = sent.polarity_scores(title)
        return print(sentiment_score)


class Yahoo:
    def __init__(self,ticker,start_date,end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = None
        self.df = None
        self.var = None
        self.volatility = None
        self.combined_df = pd.DataFrame()
    def get_historical_data(self):
        self.stock_data = yf.download(self.ticker,start=self.start_date, end=self.end_date)
    def print_data(self):
        print("\nStock:",self.ticker,self.stock_data)
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
        #useful for plotting possible
        fig = go.Figure(data=[go.Candlestick(x=self.df['Date'],
                                             open=self.df['Open'], high=self.df['High'],
                                             low=self.df['Low'], close=self.df['Close'])
                              ])
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.show()

    def summarize(self):
        # useful
        self.var = (self.df.describe())
        # print(self.var)
        return self.var

    def calculate_volatlity(self):
        #useful calcs volatility
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


class Statistics_Test:
    def test_correlation(self,stock_returns,sentiment_scores,alpha=.05):
        '''alpha significance level is .05 or 5%'''
        correlation,p_value = stats.pearsonr(stock_returns,sentiment_scores)
        if p_value < alpha:
            interpretation = "There is a significant correlation between stock returns and sentiments scores"
        else:
            interpretation = "There is not a significant correlation between stock returns and sentiment scoers"

        return correlation,p_value,interpretation

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
    '''
    weights = [1 / 20] * 19
    df = pd.DataFrame()
    asset = Asset('CE copy.csv')
    new_dfs = asset.create_dataframe()
    df['Date'] = new_dfs['Date']

    for filename in os.listdir('Berkshire/'):
        if filename.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join('Berkshire/', filename)
            Ass = Asset(file_path)
            new_df = Ass.create_dataframe()
            df1_close = Ass.close_column(new_df, ["Date", "Close"])

            filename_without_extension = os.path.splitext(filename)[0]
            df1_close.rename(columns={"Close": filename_without_extension}, inplace=True)

            df = df.merge(df1_close, on="Date", how="left")

    print(df)
    portfolio_std = df.iloc[:, 1:].std().mean()
    print(portfolio_std)

    df_copy = df.copy()
    new_frame = Ass.calculate_return(df_copy, weights)
    print(new_frame)
    #print(df_copy)
    
    '''
    # If you want to change the Pandas options to always display the entire DataFrame:
    '''
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df_copy)  # Or simply write "df" in the terminal

    # Resetting the Pandas options to the default behavior (truncate large DataFrames)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    '''
    '''
    '''
    ticker = "SPY"
    start_date = "2016-01-01"
    end_date = "2022-12-31"

    data_fetcher = Yahoo(ticker, start_date, end_date)
    data_fetcher.get_historical_data()
    data_fetcher.print_data()

    # Assets_1.summarize(df)
    # print(Assets_1)

    reddit_analysis = Reddit_Analysis()
    reddit_analysis.reddit_praw()
    reddit_analysis.clean_reddit()
    reddit_analysis.title_analysis()


if __name__ == '__main__':
    main()
