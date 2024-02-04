import robotrader as rt
import argparse
import logging
import yaml
import run_strategies as run
import pandas as pd
from datetime import date
from datetime import datetime
import pandas_market_calendars as mcal
from PostgresWorker import PostgresWorker
from BasketTrader import BasketTrader
import numpy as np


def basket_main():
    logging.basicConfig(
        level=logging.DEBUG,
        filename="logfile",
        filemode="a+",
        format="%(asctime)-15s %(levelname)-8s %(message)s")

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to the YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        # Load the YAML data into a Python object
        data = yaml.safe_load(f)
        # Now you can access the data stored in the YAML file
        strategy = data["strategy"]
        tickers = data["tickers"]
        indicators = data["indicators"]
        api_key = data["api_key"]
        redirect_url = data["redirect_url"]
        token_path = data["token_path"]
        account_id = data["account_id"]

        today = date.today()
        nyse = mcal.get_calendar('NYSE')

        if nyse.valid_days(start_date=today, end_date=today).to_list():
            trader = BasketTrader()
            trader.construct_prod_robot(
                api_key, redirect_url, token_path, account_id)
            trader.get_roboclient()
            trader.get_prod_positions()
            trader.get_prod_signals(tickers, indicators)
            trader.run_basket_trader(max_purchase=1000, testing=False, etl=True)
            df_orders = pd.DataFrame.from_dict(
                trader.order_book, orient='index').reset_index()
            trader.df_positions.to_csv("~/Documents/dan_trades/algo_trader/tables/production/position.csv")
            trader.df_signals.to_csv("~/Documents/dan_trades/algo_trader/tables/production/signals.csv")
            df_orders.to_csv("~/Documents/dan_trades/algo_trader/tables/production/orders.csv")


if __name__ == "__main__":
    basket_main()
