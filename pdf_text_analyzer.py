from collections import Counter

import pdfplumber
import re

from matplotlib import pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer


class TextAnalyzer:
    def __init__(self, file):
        self.filtered = []
        self.tokens = []
        self.file = file
        self.text = ""

    def extract_text(self):
        '''Function: extract text
        Does: uses pdf plumber to pull all text from pdfs
        :return:
        '''
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

    def analyze_text(self):
        word_freq = Counter(self.filtered)
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = analyzer.polarity_scores(self.text)
        return sentiment_scores

    def analyze(self):
        self.extract_text()
        self.preprocess_text()
        self.tokenize_text()
        self.remove_stop()
        return self.analyze_text()


    def plot(self,data):
        sent_values = list(data.values())
        sent_keys = list(data.keys())
        plt.bar(sent_keys,sent_values)
        plt.show()