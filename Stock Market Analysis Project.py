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
import time
from datetime import datetime

SPY = "SPY.csv"
AAPL_PDF = 'AAPL 10-K'
NVDA_PDF = 'NVDA 10-K.pdf'
PTON_PDF = 'PTON 10-K.pdf'
META_PDF = 'META 10K.pdf'

class Reddit_Analysis:
    def __init__(self):
        self.reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent,username=username, password=password)
        self.analyzer = SentimentIntensityAnalyzer()
        self.new_titles = []
        self.top_titles = []
        self.tickers = []
        self.clean_list = []
        self.top_posts_daily = []
    def reddit_praw(self):
        # Get top 30 posts and then get sent score and compare to return of the 30 days
        subreddit = self.reddit.subreddit('wallstreetbets')
        #new_subreddit = subreddit.new(limit=30)
        top_subreddit = subreddit.top(time_filter="month",limit=1761)
        '''
        for submission in new_subreddit:
            title = submission.title
            title_words = title.split()
            self.new_titles.append(title_words)
        '''

        for submission in top_subreddit:
            title = submission.title
            title_words = title.split()
            self.top_titles.append(title_words)
        return print(self.top_titles)#, print(self.top_titles)

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

        for title in self.top_titles:
            clean_title = emoji_pattern.sub('', ' '.join(title))
            self.clean_list.append(clean_title)

        return print(self.clean_list)

    def get_top_daily_posts(self, returns_df):
        # Source https://www.geeksforgeeks.org/python-time-mktime-method/
        # Extract unique dates from returns DataFrame
        unique_dates = returns_df.index.date.unique()
        top_posts_data = []

        # iterate over each unique date
        for date in unique_dates:
            # convert the date to Unix timestamp
            start_timestamp = time.mktime(date.timetuple())
            # end timestamp is start_timestamp + 24*60*60
            end_timestamp = start_timestamp + 24 * 60 * 60

            # get top posts for the day
            top_posts = list(self.reddit.subreddit('wallstreetbets').submissions(start=start_timestamp, end=end_timestamp))

            # sort the posts by score and take the first one
            top_post = sorted(top_posts, key=lambda post: post.score, reverse=True)[0]

            # append the post's data to the list
            top_posts_data.append(
                [top_post.title, top_post.score, top_post.id, top_post.subreddit, top_post.url, top_post.num_comments,
                 top_post.selftext, top_post.created])

        # Create a DataFrame from the top posts data
        top_posts_df = pd.DataFrame(top_posts_data,
                                    columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body',
                                             'created'])

        # Convert the 'created' column to datetime
        top_posts_df['created'] = pd.to_datetime(top_posts_df['created'], unit='s')

        return top_posts_df

    def title_analysis(self):
        self.sent_score = []
        for title in self.clean_list:
            sentiment_score = self.analyzer.polarity_scores(title)
            self.sent_score.append(sentiment_score)
        return self.sent_score
    def plot_sentiment(self):

        sentiment_df = pd.DataFrame(self.sent_score, columns=['compound', 'neg', 'neu', 'pos'])
        sentiment_df['post_number'] = range(1, len(sentiment_df) + 1)

        plt.figure(figsize=(10, 6))
        sns.regplot(x=sentiment_df['post_number'], y=sentiment_df['compound'], label='Compound',line_kws={'color':'darkblue'},scatter_kws={'color':'blue'})
        sns.regplot(x=sentiment_df['post_number'], y=sentiment_df['neg'], label='Negative',line_kws={'color':'firebrick'}, scatter_kws={'color':'red'})
        sns.regplot(x=sentiment_df['post_number'], y=sentiment_df['neu'], label='Neutral',line_kws={'color':'coral' }, scatter_kws={'color':'orange'})
        sns.regplot(x=sentiment_df['post_number'], y=sentiment_df['pos'], label='Positive',line_kws={'color':'darkgreen'},scatter_kws={'color':'green'})
        plt.xlabel('Days')
        plt.ylabel('Sentiment Score (Pos,Neg,Neu)')
        plt.title('Sentiment Score Trend')
        plt.legend()
        plt.show()
        # Calculate correlation coefficients
        corr_compound, _ = stats.pearsonr(sentiment_df['post_number'], sentiment_df['compound'])
        corr_neg, _ = stats.pearsonr(sentiment_df['post_number'], sentiment_df['neg'])
        corr_neu, _ = stats.pearsonr(sentiment_df['post_number'], sentiment_df['neu'])
        corr_pos, _ = stats.pearsonr(sentiment_df['post_number'], sentiment_df['pos'])
        return print('\nNegative Corr:',corr_neg,'\nNeutral Corr:',corr_neu,'\nPositive Corr:',corr_pos)


    def combined_df(self, sentiment_scores, final_return):
        sentiment_df = pd.DataFrame(sentiment_scores, columns=['compound', 'neg', 'neu', 'pos'])
        # reset the index to match with sentiment_df
        final_return.reset_index(drop=True, inplace=True)
        # Combine sentiment scores and portfolio returns into a single DataFrame
        combined_df = pd.concat([sentiment_df, final_return], axis=1)
        return combined_df

    def plot_sentiment_vs_returns(self, sentiment_scores, final_return):
        sentiment_df = pd.DataFrame(sentiment_scores, columns=['compound', 'neg', 'neu', 'pos'])
        # reset the index to match with sentiment_df
        final_return.reset_index(drop=True, inplace=True)
        # Combine sentiment scores and portfolio returns into a single DataFrame
        combined_df = pd.concat([sentiment_df, final_return], axis=1)

        # Plot correlation matrix
        sns.regplot(data=combined_df, x='compound', y='Portfolio_Return')
        sns.regplot(data = combined_df,x='neg', y='Portfolio_Return', label='Negative',
                    line_kws={'color': 'firebrick'}, scatter_kws={'color': 'red'})
        sns.regplot(data=combined_df,x='neu', y='Portfolio_Return', label='Neutral', line_kws={'color': 'coral'},
                    scatter_kws={'color': 'orange'})
        sns.regplot(data=combined_df,x='pos', y='Portfolio_Return', label='Positive',
                    line_kws={'color': 'darkgreen'}, scatter_kws={'color': 'green'})
        plt.xlabel('Sentiment Score (Neg,Neu,Pos,Compound)')
        plt.ylabel('Portfolio Return')
        plt.title('Sentiment Scores vs Portfolio Returns')
        plt.show()


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

def calculate_return(port_data):
    port_return = pd.DataFrame()
    port_return['Date'] = port_data['Date']
    for stock in port_data.columns[1:]:
        port_return[stock  + '_Return'] = port_data[stock].pct_change().fillna(0)
    return port_return

def portfolio_return(dataframe, weights):
    final_return = pd.DataFrame()
    final_return['Date'] = dataframe['Date']
    final_return['Portfolio_Return'] = dataframe.iloc[:, 1:].mul(weights).sum(axis=1)
    return final_return

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

def sharpe_ratio(df, rf, std):
    df['Sharpe_Ratio'] = df['Portfolio_Return'].apply(lambda x: (x - rf) / std)
    return df



def calculate_portfolio_std(dataframe, weights):
    """
    Calculates the portfolio standard deviation using a pandas DataFrame of different stocks and their returns.
    Args: dataframe (pandas.DataFrame): DataFrame containing stock returns.
    Each column represents a stock and each row represents a return value.
    Returns: float: Portfolio standard deviation.
    """
    # Calculate the covariance matrix
    covariance_matrix = dataframe.cov(numeric_only=True)

    # Convert weights to NumPy array
    weights = np.array(weights)

    # Calculate the portfolio variance
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))

    # Calculate the portfolio standard deviation
    portfolio_std = np.sqrt(portfolio_variance)
    return portfolio_std


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
        # data_fetcher.print_data()
        data = data_fetcher.get_data_as_dataframe()
        df_close = close_column(data, ["Date", "Close"])
        df_close = df_close.rename(columns={"Close": ticker})
        closing_df = closing_df.merge(df_close, on="Date", how="left")
    #print(closing_df)

    # weight of each stock in portfolio
    weights = [1 / 20] * 20
    #print(weights)
    closing_copy = closing_df.copy()
    port_return = calculate_return(closing_copy)
    #print(port_return)

    final_return = (portfolio_return(port_return, weights))
    #print(final_return)

    std = calculate_portfolio_std(port_return, weights)
    #print(std)

    rf = 1.74
    #final_return = (sharpe_ratio(final_return, rf, std))
    print(len(final_return))
    print(final_return)
    reddit_analysis = Reddit_Analysis()
    reddit_analysis.reddit_praw()
    reddit_analysis.clean_reddit()
    sent_scores = reddit_analysis.title_analysis()
    print(sent_scores)
    '''Results: Negative Corr: -0.05421956596167326,Neutral Corr: 0.05401874549094921,Positive Corr: 9.271238516046039e-05'''
    print(reddit_analysis.combined_df(sent_scores,final_return))
    #reddit_analysis.plot_sentiment()
    reddit_analysis.plot_sentiment_vs_returns(sent_scores, final_return)



if __name__ == '__main__':
    main()



'''Unused Code

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

'''