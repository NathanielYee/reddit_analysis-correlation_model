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

SPY = "SPY.csv"
AAPL_PDF = 'AAPL 10-K'
NVDA_PDF = 'NVDA 10-K.pdf'
PTON_PDF = 'PTON 10-K.pdf'


class TextAnalyzer:
    def __init__(self, file):
        self.filtered = []
        self.tokens = []
        self.file = file
        self.text = ""

    def extract_text(self):
        with pdfplumber.open(self.file) as pdf:
            for page in pdf.pages:
                self.text += page.extract_text()

    def preprocess_text(self):
        # Remove unwanted sections, convert to lowercase, remove punctuation
        # Add any other necessary preprocessing steps

        # Remove table of contents
        self.text = re.sub(r'Table of Contents.*?\n', '', self.text, flags=re.IGNORECASE | re.DOTALL)

        # Convert to lowercase
        self.text = self.text.lower()

        # Remove punctuation
        self.text = re.sub(r'[^\w\s]', '', self.text)

    def tokenize_text(self):
        self.tokens = word_tokenize(self.text)

    def remove_stop(self):
        stop_words = set(stopwords.words('english'))
        self.filtered = [token for token in self.tokens if token.lower() not in stop_words]

    def anaylze_text(self):
        word_freq = Counter(self.filtered)
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(self.text)
        return print(sentiment_scores)

    def analyze(self):
        self.extract_text()
        self.preprocess_text()
        self.tokenize_text()
        self.remove_stop()
        self.anaylze_text()


class Asset:
    '''
    def __init__(self, *components):
        self.components = []
    '''

    def __init__(self, file):
        self.file = file
        # self.df = None
        # self.var = None

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

    def summarize(self):
        self.var = (self.df.describe())
        # print(self.var)
        return self.var

    def sharpe_ratio(self):
        self.sd = self.var.loc['std']  # standard deviation
        # print(self.sd)
        return self.sd

    def process_file(self):
        self.create_dataframe()
        self.summarize()
        self.sharpe_ratio()


def main():
    '''
    Assets_1 = Asset(SPY)
    Assets_1.process_file()
    summary = Assets_1.summarize()
    print(summary)

    stdev = Assets_1.sharpe_ratio()
    print(stdev)
    '''
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
if __name__ == '__main__':
    main()
