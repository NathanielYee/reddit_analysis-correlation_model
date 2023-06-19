
# class Asset:
#     '''
#     def __init__(self, *components):
#         self.components = []
#     '''
#
#     def __init__(self, file):
#         self.volatility = None
#         self.file = file
#         self.df = None
#         self.var = None
#         self.combined_df = pd.DataFrame()
#
#     '''
#     def __repr__(self,*args):
#         # 0th position is always a str represented by the date
#         attributes = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
#         return f"Asset({attributes})"
#     '''
#
#     def create_dataframe(self):
#         self.df = pd.read_csv(self.file)
#         return self.df
#
#     def candlestick(self):
#         fig = go.Figure(data=[go.Candlestick(x=self.df['Date'],
#                                              open=self.df['Open'], high=self.df['High'],
#                                              low=self.df['Low'], close=self.df['Close'])
#                               ])
#         fig.update_layout(xaxis_rangeslider_visible=False)
#         fig.show()
#
#     def summarize(self):
#         self.var = (self.df.describe())
#         # print(self.var)
#         return self.var
#
#     def calculate_volatlity(self):
#         returns = self.df['Close'].pctchange().dropna()
#         self.volatility = np.sqrt(252) * returns.std()
#         return self.volatility
#
#     def calculate_return(self, port_data, weights):
#         port_return = pd.DataFrame()
#         for stock in port_data.columns[1:]:
#             port_return[stock + '_Return'] = port_data[stock].pct_change().fillna(0)
#
#         port_return['Portfolio_Return'] = port_return.iloc[:, 1:].mul(weights).sum(axis=1)
#         return port_return
#
#     def standard_deviation(self):
#         self.sd = self.var.loc['std']  # standard deviation
#         # print(self.sd)
#         return self.sd
#
#     def process_file(self):
#         self.create_dataframe()
#         self.candlestick()
#         self.summarize()


# def close_column(dataframe, column_name):
#     """ Trying to add all the close columns into one panda dataframe and return it """
#     df1_close = dataframe[column_name].copy()
#     return df1_close
#
#
# def create_closing(file):
#     # Initialize the first pandas dataframe that takes in all the closing prices of each stock in the portfolio
#     closing_df = pd.DataFrame()
#     asset = Asset(file)
#     new_dfs = asset.create_dataframe()
#     # creating the first column for the pandas to help merge each of the pd df based on the date column
#     closing_df['Date'] = new_dfs['Date']
#     return closing_df
