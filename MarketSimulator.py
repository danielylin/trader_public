import utils as ut
import pandas as pd
import BasketAlgo as ba
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

pd.options.mode.chained_assignment = None  # default='warn'


class MarketSimulator(object):
    """This class simulates trades."""

    def __init__(self,
                 df_trades,
                 data_source,
                 ticker,
                 start_port_val,
                 shares=None,
                 commission=None,
                 impact=None,
                 client=None):

        self.df_trades = df_trades
        self.data_source = data_source
        self.ticker = ticker
        self.start_port_val = start_port_val
        self.commission = commission
        self.impact = impact
        self.shares = shares
        self.start_date = df_trades.index.min()
        self.end_date = df_trades.index.max()

        if self.data_source in ["td", "yf"]:
            if self.data_source == "td":
                client = ut.get_ameritrade_client()
            else:
                client = None

            self.df_prices = ut.get_data(
                tick=self.ticker,
                time_interval="day",
                source=data_source,
                start_date=self.start_date,
                end_date=self.end_date,
                client=client)
        else:
            self.df_prices = pd.read_csv(data_source, index_col="date")

    def simulate_trades(self, stop_loss_level=None, trailing=False):
        colnames = ['close', 'cash', self.ticker, 'portval']

        if self.shares is not None:
            self.df_trades.loc[:, 'trade'] = self.df_trades['trade']*self.shares

        self.df_portval = pd.DataFrame(
            0, index=self.df_trades.index, columns=colnames)

        self.df_portval.loc[self.start_date, 'cash'] = self.start_port_val

        prev_shares = 0
        prev_cash = self.start_port_val
        highest_price = None

        for index, row in self.df_trades.iterrows():
            close_price = self.df_prices.loc[index, 'close'].item()

            # Check if stop loss should be triggered
            if prev_shares > 0 and stop_loss_level is not None:
                if highest_price is None or (trailing and close_price > highest_price):
                    highest_price = close_price

                if close_price <= (1 - stop_loss_level) * highest_price:
                    # Sell all shares
                    row['trade'] = -prev_shares

            self.df_portval.loc[index, 'cash'] = prev_cash - row['trade']*close_price
            self.df_portval.loc[index, self.ticker] = prev_shares + row['trade']
            self.df_portval.loc[index, 'close'] = close_price

            self.df_portval.loc[index, 'portval'] = self.df_portval.loc[index, 'cash'] + self.df_portval.loc[index, self.ticker]*close_price

            prev_shares = self.df_portval.loc[index, self.ticker]
            prev_cash = self.df_portval.loc[index, 'cash']


def generate_portfolio_stats(portvals, n, ticker=None):
    """Generates portfolio statistics.

    params:
        portvals: dataframe of portfolio values
        n: sharpe ratio scaling factor e.g. normalized_sr = sr/sqrt(n)

    return type: a dataframe with portfolio stats
    """
    cum_ret = portvals.iloc[-1] / portvals.iloc[0]-1
    df_daily_returns = (portvals/portvals.shift(1))-1
    df_daily_returns.iloc[0] = 0
    df_daily_returns = df_daily_returns[1:]
    avg_daily_ret = df_daily_returns.mean(axis=0)
    std_daily_ret = df_daily_returns.std(axis=0)
    sharpe_ratio = avg_daily_ret/std_daily_ret*(n**.5)

    min_date = min(df_daily_returns.index)
    max_date = max(df_daily_returns.index)
    df_stats = pd.DataFrame(data=[
        ticker,
        cum_ret,
        avg_daily_ret,
        std_daily_ret,
        sharpe_ratio,
        min_date,
        max_date
        ]).transpose()
    df_stats.columns = [
            "ticker",
            "cum_ret",
            "avg_daily_ret",
            "std_daily_ret",
            "sharpe_ratio",
            "start_date",
            "end_date"]

    return df_stats

if __name__ == "__main__":
    pass


