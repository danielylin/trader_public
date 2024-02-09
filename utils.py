"""
Utils contains functions used by all modules

Usage::
import utils as ut
"""

from tda.auth import easy_client
import yfinance as yf
import pandas as pd
import datetime as dt
from datetime import timedelta, datetime


def get_data(tick: str,
             time_interval: str,
             source: str,
             start_date="2021-01-01",
             end_date="2022-01-01",
             period: str = None,
             client=None) -> pd.DataFrame:
    """
    params:
        tick: string stock ticker
        time_interval: granularity of time data, depends on your API
            day: 1d (yf), day (td)
        source: yf, td currently supported
        start_date: start date of the data
        end_date: end date of the data
        period = some etl clients accept 6mo
        client = the etl client if manual (td)
    return: pandas dataframe with column name = tick and equity price as values

    description: get_data returns the equity prices in single dataframe column
    """

    col_names = ["symbol", "open", "high", "low", "close", "volume"]

    if (isinstance(start_date, str)) or (isinstance(end_date, str)):
        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = dt.datetime.strptime(
            end_date, "%Y-%m-%d").date()

    if (isinstance(start_date, datetime)) or (isinstance(end_date, datetime)):
        start_date = start_date.date()
        end_date = end_date.date()

    start_date_buff = start_date - timedelta(days=7)
    end_date_buff = min(end_date + timedelta(days=7), datetime.now().date())

    # Read Yahoo API yfinance
    if source == "yf":
        if time_interval == "day":
            time_interval = "1d"

        ticker = yf.Ticker(tick)

        if period is not None:
            df_prices = ticker.history(
                period=period,  interval=time_interval, auto_adjust=False)
        else:
            df_prices = ticker.history(
                start=start_date_buff, end=end_date_buff,
                interval=time_interval, auto_adjust=False)
        if df_prices.empty:
            return None

        df_prices = df_prices.sort_index()
        df_prices.index = df_prices.index.date
        df_prices.index.name = "date"
        df_prices["symbol"] = tick
        df_prices = df_prices[
            ["symbol", "Open", "High", "Low", "Close", "Volume"]]

        df_prices = df_prices.rename(
            columns=dict(zip(df_prices.columns, col_names)))

        df_prices = df_prices[(
           (df_prices.index >= start_date) &
           (df_prices.index <= end_date))]

        return df_prices

    elif source == "td":
        if time_interval.lower() == "day":

            if period is not None:
                if period == "6mo":
                    start_date = dt.datetime.today() - timedelta(days=185)
                else:
                    start_date = dt.datetime.today() - timedelta(days=252)
                end_date = dt.datetime.today()

            start_date = datetime.combine(start_date, datetime.min.time())
            end_date = datetime.combine(end_date, datetime.min.time())

            if client is None:
                client = get_ameritrade_client() #  Not available publically due to PII. Define your own callback here.

            df_json = client.get_price_history_every_day(
                tick,
                start_datetime=start_date,
                end_datetime=end_date).json()

            df = pd.DataFrame.from_dict(df_json)
            df_prices = pd.concat(
                [df.drop(["candles"], axis=1), df["candles"].apply(pd.Series)],
                axis=1)
            df_prices["datetime"] = pd.to_datetime(
                df_prices["datetime"], unit="ms")
            df_prices["date"] = pd.to_datetime(
                pd.to_datetime(df_prices["datetime"]).dt.date)
            df_prices = df_prices.set_index("date")
            df_prices = df_prices[
                ["symbol", "open", "high", "low", "close", "volume"]]
            df_prices = df_prices[
                (df_prices.index >= start_date) &
                (df_prices.index <= end_date)]
            return df_prices


def get_sp500():
    """Gets the companies from the S&P500"""
    payload = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df_tickers = payload[0]
    tickers = df_tickers["Symbol"].values.tolist()
    tickers = [ticker.replace(".", "-") for ticker in tickers]
    return tickers

if __name__ == "__main__":
    pass
