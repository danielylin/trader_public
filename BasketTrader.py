import robotrader as rt
import argparse
import logging
import yaml
import run_strategies as run
import math
import pandas as pd
from datetime import date
from datetime import datetime
import pandas_market_calendars as mcal
from PostgresWorker import PostgresWorker
import numpy as np

class BasketTrader(object):
    """This class creates an object that runs the basket algo strategy.
    """
    def __init__(self):
        self.api_key = None
        self.redirect_url = None
        self.token_path = None
        self.account_id = None

    def construct_prod_robot(self, api_key, redirect_url, token_path,
                             account_id):
        self.api_key = api_key
        self.redirect_url = redirect_url
        self.token_path = token_path
        self.account_id = account_id

    def get_roboclient(self):
        self.robo_basket = rt.RoboTraderTD(
            api_key=self.api_key,
            redirect_url=self.redirect_url,
            token_path=self.token_path,
            account_id=self.account_id)
        self.robo_basket.get_client()

    def get_prod_positions(self):
        self.df_positions = self.robo_basket.get_positions()

    def get_prod_signals(self, tickers, indicators):
        self.df_signals = run.basket_iterate_ticks(tickers, indicators, "td")

    def set_test_positions(self, df_test_positions):
        self.df_positions_testing = df_test_positions

    def set_test_signals(self, df_test_signal):
        self.df_signals_testing = df_test_signal

    def initialize_etl(self,
                       database="postgres",
                       user="daniellin",
                       password="",
                       host="127.0.0.1",
                       port="5432",
                       trader=None):
        self.sql_worker = PostgresWorker(
            database, user, password, host, port, trader)
        self.sql_worker.connect()

    def run_basket_trader(self,
                          max_purchase=1000,
                          testing=True,
                          etl=False,
                          repurchase=False):
        if not testing:
            df_positions = self.df_positions
            df_signals = self.df_signals
            non_marginable_funds = self.robo_basket.get_non_marginable_funds()
            buff = 5000
        else:
            df_positions = self.df_positions_testing
            df_signals = self.df_signals_testing
            non_marginable_funds = df_positions[
                df_positions["symbol"] == "MMDA1"]["longQuantity"].item()
            buff = 0

        signals = df_signals[
            df_signals["current_date"] == df_signals["signal_date"]]
        positions = np.unique(df_positions["symbol"])

        self.order_book = dict()

        for index, row in signals.iterrows():
            ticker = row["symbol"]
            df_tick = df_positions[df_positions["symbol"] == ticker]

            # Initialize vars to 0.
            current_shares = 0
            purchase_price = 0
            num_shares = 0
            order_type = None
            new_long_quantity = 0
            new_short_quantity = 0
            long_quantity = 0
            short_quantity = 0
            average_price = 0
            final_shares = 0
            purchasing_power = non_marginable_funds-buff

            # Get current shares.
            if ticker in positions:
                long_quantity = df_tick["longQuantity"].item()
                short_quantity = df_tick["shortQuantity"].item()
                average_price = abs(df_tick["averagePrice"].item())

                if short_quantity > 0:
                    current_shares = -short_quantity
                if long_quantity > 0:
                    current_shares = long_quantity
            else:
                current_shares = 0

            # First we'll handle buy cases.
            if (row["position"] > 0 and current_shares < 0):
                final_shares = -1*current_shares
                order_type = "buy"
                purchase_price = row["current_price"]+0.1

            if (row["position"] > 0 and (current_shares == 0 or repurchase)):
                order_type = "buy"
                purchase_price = row["current_price"]+0.1
                final_shares = math.floor(max_purchase/purchase_price)

             # Now let's handle sell cases.
            if (row["position"] < 0 and current_shares > 0):
                final_shares = -1*current_shares
                order_type = "sell"
                purchase_price = row["current_price"]-0.1
            if (row["position"] < 0 and (current_shares == 0 or repurchase)):
                order_type = "sell"
                purchase_price = row["current_price"]-0.1
                final_shares = -math.floor(max_purchase/purchase_price)

            # Check if have money
            if order_type is not None:
                num_shares = final_shares-current_shares

            if purchasing_power < abs(num_shares*purchase_price):
                order_type = None
                num_shares = 0

            # Execute the trade.
            if num_shares != 0:
                if testing is False:
                    self.robo_basket.purchase_equity(
                        ticker, purchase_price, abs(num_shares), order_type)

            if abs(num_shares) > 0:
                message = "{} {} shares of {} at {}".format(
                    order_type, abs(num_shares), ticker, purchase_price)
            elif num_shares == 0:
                message = "Already own of {} of {} at {}".format(
                    final_shares, ticker, purchase_price)
            else:
                message = "An error has occured."
            logging.info(message)

            if order_type == "buy":
                new_long_quantity = long_quantity + num_shares
                new_short_quantity = 0
            elif order_type == "sell":
                new_short_quantity = short_quantity - num_shares
                new_long_quantity = 0
            else:
                new_short_quantity = short_quantity
                new_long_quantity = long_quantity

            current_date = max(df_signals["current_date"])

            self.order_book[ticker] = [
                current_date,
                long_quantity,
                short_quantity,
                average_price,
                num_shares,
                purchase_price,
                (non_marginable_funds - buff),
                new_long_quantity,
                new_short_quantity]

        trade_client = None if testing is True else self.robo_basket

        if etl is True:
            self.initialize_etl(trader=trade_client)
            self.sql_worker.connect()
            if bool(self.order_book):
                # self.sql_worker.export_requested_orders(
                #     "requested_orders",
                #     self.order_book,
                #     write_type="append"
                # )

                # self.sql_worker.export_filled_orders(
                #     "filled_orders",
                #     datetime.combine(date.today(), datetime.min.time()),
                #     write_type="append"
                # )
                if trade_client is not None:
                    self.sql_worker.export_positions(
                        "positions",
                        write_type="append"
                    )