import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

class Asset:
    def __init__(self, *components):
        self.components = []

    def read(self,file):
        df = pd.read_csv(file)
        return df
