import pandas as pd
import numpy as np

import MarketSimulator as sim
import BasketAlgo as basket
import matplotlib.pyplot as plt

import utils as ut
import warnings
import dca
import logging

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.options.mode.chained_assignment = None


def get_latest_basket_signal_single(df_prices: pd.DataFrame,
                                    indicators: str,
                                    tick: str) -> list:
    """This function returns the latest basket signal for a single ticker.

    params:
        df_prices: dataframe of prices returned from the get_data method,
        gererally each row represents one unit of time
        indicators: list of indicators to evaluate signal
        tick: the ticker to evaluate

    returns:
        res: An array of values with regards to the action to take.
    """
    indicators = ["bb", "rsi", "aroon"]
    basket_strat = basket.BasketAlgo(
        indicators=indicators,
        ticker=tick,
        df_prices=df_prices
    )
    basket_strat.run_strategy()

    df_signals = basket_strat.df_signals
    signal_date = df_signals.sort_index(ascending=False).head(1).index.item()
    df_indicators = df_signals[indicators]
    df_gt = df_indicators.gt(0)
    df_lt = df_indicators.lt(0)
    df_indicators["bull_signal"] = df_gt.apply(
        lambda x: x.index[x].tolist(), axis=1)
    df_indicators["bear_signal"] = df_lt.apply(
        lambda x: x.index[x].tolist(), axis=1)
    bull_signal = df_indicators.loc[signal_date]["bull_signal"]
    bear_signal = df_indicators.loc[signal_date]["bear_signal"]
    price_at_signal = df_signals.loc[signal_date][tick]
    res = [df_signals.index[-1], tick, df_signals["position"][-1], signal_date,
           bull_signal, bear_signal, price_at_signal, df_signals[tick][-1]]
    return res


def basket_iterate_ticks(tickers, indicators, data_source, *args):
    """Iterates through tickers and returns all actions.
    params:
        tickers: an array of tickers
        indicators: list of indicators to evaluate
        data_source: td or yf

    returns: df positions for all tickers
    """
    positions_list = []

    if data_source == "td":
        client = ut.get_ameritrade_client()
    else:
        client = None

    for tick in tickers:
        df_prices = ut.get_data(
            tick=tick,
            time_interval="day",
            source=data_source,
            period="6mo",
            client=client)
        res = get_latest_basket_signal_single(df_prices, indicators, tick)
        positions_list.append(res)

    df_positions = pd.DataFrame(positions_list, columns=[
        "current_date",
        "symbol",
        "position",
        "signal_date",
        "bull_signal",
        "bear_signal",
        "price_at_signal",
        "current_price"])

    return df_positions


def get_latest_signal(tickers, data_source, strategy, indicators):
    """A wrapper for generating positions list off generic names
    params:
        tickers: an array of tickers
        data_source: td or yf
        strat: currently baskets
        indicators: list of indicators

    returns:
        df_positions: latest signal for the each ticker
    """
    if strategy == "basket":
        df_positions = basket_iterate_ticks(tickers, indicators, data_source)
        return df_positions
    else:
        print(("Error: no signal simulated. Please check if {} exists")
              .format(strategy))
        return None


def write(df: pd.DataFrame, filepath: str, incremental: bool = True):
    """writes to a file, overwritten or incremental
    params:
        df: pandas dataframe
        filepath: export path, incremental = write type
    """
    if incremental:
        df.to_csv(filepath, mode="a", header=False)
    else:
        df.to_csv(filepath)


def get_trades_from_strat(df_prices: pd.DataFrame,
                          indicators: list,
                          strategy: str = "basket",
                          shares: int = 1000):
    """Gets the trades given a strategy.

    params:
        df_prices: pandas dataframe of prices
        indicators: list of indicators
        strategy: trading strategy (only basket currently supported)
        shares: number of shares to purchase
    returns:
        df_signals: the signals from the indicators
        df_simulated_trades: the simulated trades
    """
    if strategy == "basket":
        # tick = df_prices.head(1)["symbol"].item()
        tick = df_prices["symbol"].iloc[0]
        basket_strat = basket.BasketAlgo(
            indicators=indicators,
            ticker=tick,
            df_prices=df_prices
        )
        basket_strat.run_strategy()
        return basket_strat.df_signals, basket_strat.df_trades
    elif strategy == "dca":
        df_signals = pd.DataFrame([-1])
        tick = df_prices.columns.item()
        df_simulated_trades = dca.create_dca_trades_df(
            df_prices, tick, 5, shares)
        return df_signals, df_simulated_trades
    else:
        print(("Error: no signal simulated. Please check if {} exists")
              .format(strategy))
        return None


def test_strategies(indicators: list,
                    start_date: str,
                    end_date: str,
                    tickers: list,
                    interval: str,
                    start_val: int,
                    impact: float,
                    commission: float,
                    data_source: str,
                    strategy: str = "basket",
                    plot_path: str = "plots",
                    max_shares: bool = True,
                    shares: int = 1000,
                    benchmark_shares: int = 1000):
    """Generates table and graphs to backtest strategies.
    params:
        - indicators: list of indicators
        - start_date and end_date of testing period
        - tickers is a list of equities
        - interval is the date interval to test
        - start_val is starting portfolio cash
        - impact is how your purchase impacts the price
        - commission is the price per trade
        - shares is the the default shares you trade
        - data_source can be yf (yahoo finance) or td (ameritrade)
        - strategy: trading strategy to implement
        - plot path: string of file path
        - max_shares: whether to max out the shares

    returns:
        ticker_strategy: tuple of stats
    """
    ticker_strategy = {}
    logging.basicConfig(filename='log.txt', level=logging.INFO)

    for tick in tickers:
        logging.info(f"Running algo on {tick}.")
        if data_source == "td":
            client = ut.get_ameritrade_client()
        else:
            client = None

        df_prices = ut.get_data(
            tick=tick,
            time_interval=interval,
            source=data_source,
            start_date=start_date,
            end_date=end_date,
            client=client)

        if df_prices is None or len(df_prices) < 1:
            continue

        # Run basket strategy.
        df_signals, df_simulated_trades = get_trades_from_strat(
            df_prices, indicators=indicators, strategy=strategy, shares=shares)

        if max_shares:
            n = int(np.floor(start_val/df_prices.head(1)["close"].item()))
            shares = n
            benchmark_shares = n

        marketsim = sim.MarketSimulator(
            df_trades=df_simulated_trades,
            data_source=data_source,
            ticker=tick,
            start_port_val=start_val,
            shares=shares
        )
        marketsim.simulate_trades()
        df_portvals = marketsim.df_portval.sort_index()
        df_portval_stats = sim.generate_portfolio_stats(
            df_portvals["portval"], 252, tick)

        # Run benchmark.
        df_benchmark_trades = generate_benchmark_trades(
            df_prices, tick, benchmark_shares)

        benchmark_tick = tick
        benchmark_sim = sim.MarketSimulator(
            df_trades=df_benchmark_trades,
            data_source=data_source,
            ticker=benchmark_tick,
            start_port_val=start_val,
            shares=1
        )
        benchmark_sim.simulate_trades()
        df_benchmark_portvals = benchmark_sim.df_portval.sort_index()
        df_benchmark_stats = sim.generate_portfolio_stats(
            df_benchmark_portvals["portval"], 252, benchmark_tick)

        # Run Spy Benchmark
        df_prices_spy = ut.get_data(
            tick="SPY",
            time_interval=interval,
            source=data_source,
            start_date=start_date,
            end_date=end_date,
            client=client)

        spy_benchmark_shares = int(
            np.floor(start_val/df_prices_spy.head(1)["close"].item()))

        df_spy_benchmark_trades = generate_benchmark_trades(
            df_prices_spy, "SPY", spy_benchmark_shares)

        spy_benchmark_sim = sim.MarketSimulator(
            df_trades=df_spy_benchmark_trades,
            data_source=data_source,
            ticker="SPY",
            start_port_val=start_val,
            shares=1
        )
        spy_benchmark_sim.simulate_trades()
        df_spy_benchmark_portvals = spy_benchmark_sim.df_portval.sort_index()
        df_spy_benchmark_stats = sim.generate_portfolio_stats(
            df_spy_benchmark_portvals["portval"], 252, "SPY")

        if plot_path is not None:
            filepath = "{}{}/{}/{}_{}to{}-{}.png".format(
                plot_path,
                data_source,
                strategy,
                tick,
                start_date.replace("-", ""),
                end_date.replace("-", ""),
                data_source)

            portvals = [
                df_portvals,
                df_benchmark_portvals,
                df_spy_benchmark_portvals
            ]

            plot_comparisons(
                portvals,
                tick,
                filepath=filepath
            )
            plt.close()

        ticker_strategy[tick] = (
            df_signals, df_simulated_trades,
            df_portvals["portval"].to_frame(),
            df_benchmark_portvals["portval"].to_frame(),
            df_spy_benchmark_portvals["portval"].to_frame(),
            df_portval_stats, df_benchmark_stats, df_spy_benchmark_stats)

        logging.info(f"Successfully analyzed {tick}.")

    return ticker_strategy


def plot_comparisons(portvals,
                     tick,
                     filepath,
                     strategies: list = ["Quant", "B&H", "B&H SPY"]):
    """Saves a plot comparing 2 strategies

    params:
        portvals: list of portvals to compare
        tick: ticker evaluated
        filepath: file to save the graph
    """
    fig, ax = plt.subplots()

    for i, portval in enumerate(portvals):
        portval = portval["portval"]
        portval = portval/portval.iloc[0]

        plt.plot(
            portval.index,
            portval.values,
            label=strategies[i],
            linewidth=0.5,
            color=ut.COLOR_LIST[i]
        )

        plt.annotate("{:.2%}".format(portval.iloc[-1].item()-1),
                     xy=(portval.index[-1], portval.iloc[-1].item()),
                     xycoords=("data", "data"), size=7)

    ax.legend()
    plt.xticks(rotation=45)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (Normalized)")
    title = "{} Strategies".format(tick)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(filepath, dpi=200)


def generate_benchmark_trades(df_prices, symbol, shares):
    """Generates a trades dataframe for buy and hold strategy

    params:
        df_prices: prices returned from get_data
        symbol: the ticker,
        shares: number of shares bought for buy and hold

    returns:
        df_trades: buy and hold trades dataframe
    """
    df_trades = df_prices.copy()
    df_trades = df_trades[["close", "symbol"]]
    df_trades.columns = ["trade", "ticker"]

    df_trades["trade"] = 0
    df_trades.loc[df_trades.index.min(), "trade"] = shares

    return df_trades


def get_indicators(df_prices: pd.DataFrame, indicators: list):
    """Gets indicators for a prices dataframe.

    params:
        df_prices is a prices returned from get_data
        indicators is a list of indicators to generate signals

    returns:
        df_signals: signals dataframe
    """
    tick = df_prices["symbol"].iloc[0]
    basket_strat = basket.BasketAlgo(
        indicators=indicators,
        ticker=tick,
        df_prices=df_prices
    )
    basket_strat.run_strategy()

    return basket_strat.df_signals


def evaluate_tickers(
    tickers,
    start_date,
    end_date,
    data_source
):
    start_val = 100000
    impact = 0
    commission = 0
    impact = 0
    interval = "day"
    shares = 100
    indicators = ["bb", "rsi", "aroon"]
    max_shares = True
    benchmark_shares = 100

    strat = test_strategies(
        indicators=indicators,
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        interval=interval,
        start_val=start_val,
        impact=impact,
        commission=commission,
        shares=shares,
        data_source=data_source,
        strategy="basket",
        plot_path="plots/",
        max_shares=max_shares,
        benchmark_shares=benchmark_shares)

    df_combined_quant_stats = pd.concat(
        [tup[5] for tup in strat.values()], ignore_index=True)
    df_combined_bm_stats = pd.concat(
        [tup[6] for tup in strat.values()], ignore_index=True)
    df_combined_spy_stats = pd.concat(
        [tup[6] for tup in strat.values()], ignore_index=True)

    df_combined_bm_stats.columns = [
        'ticker',
        'cum_ret_bm',
        'avg_daily_ret_bm',
        'std_daily_ret_bm',
        'sharpe_ratio_bm',
        'start_date',
        'end_date'
    ]

    df_combined_spy_stats.columns = [
        'ticker',
        'cum_ret_spy',
        'avg_daily_ret_spy',
        'std_daily_ret_spy',
        'sharpe_ratio_spy',
        'start_date',
        'end_date'
    ]

    df_combined_stats = pd.merge(
        df_combined_quant_stats,
        df_combined_bm_stats, on=["ticker", "start_date", "end_date"])

    df_combined_stats = pd.merge(
        df_combined_stats,
        df_combined_spy_stats,
        on=["ticker", "start_date", "end_date"]
    )

    return df_combined_stats


if __name__ == "__main__":

    # tickers = ut.get_sp500()

    # start_val = 100000
    # impact = 0
    # commission = 0
    # start_date = "2021-06-01"
    # end_date = "2022-06-01"
    # impact = 0
    # interval = "day"
    # commission = 0
    # shares = 100
    # indicators = ["bb", "rsi", "aroon"]
    # max_shares = True
    # data_source = "yf"
    # benchmark_shares = 100

    # strat = test_strategies(
    #     indicators=indicators,
    #     start_date=start_date,
    #     end_date=end_date,
    #     tickers=tickers,
    #     interval=interval,
    #     start_val=start_val,
    #     impact=0,
    #     commission=0,
    #     shares=shares,
    #     data_source=data_source,
    #     strategy="basket",
    #     plot_path="plots/",
    #     max_shares=max_shares,
    #     benchmark_shares=benchmark_shares)

    # df_combined_quant_stats = pd.concat(
    #     [tup[5] for tup in strat.values()], ignore_index=True)
    # df_combined_bm_stats = pd.concat(
    #     [tup[6] for tup in strat.values()], ignore_index=True)
    # df_combined_spy_stats = pd.concat(
    #     [tup[6] for tup in strat.values()], ignore_index=True)

    # df_combined_bm_stats.columns = [
    #     'ticker',
    #     'cum_ret_bm',
    #     'avg_daily_ret_bm',
    #     'std_daily_ret_bm',
    #     'sharpe_ratio_bm',
    #     'start_date',
    #     'end_date'
    # ]

    # df_combined_spy_stats.columns = [
    #     'ticker',
    #     'cum_ret_spy',
    #     'avg_daily_ret_spy',
    #     'std_daily_ret_spy',
    #     'sharpe_ratio_spy',
    #     'start_date',
    #     'end_date'
    # ]

    # df_combined_stats = pd.merge(
    #     df_combined_quant_stats,
    #     df_combined_bm_stats, on=["ticker", "start_date", "end_date"])

    # df_combined_stats = pd.merge(
    #     df_combined_stats,
    #     df_combined_spy_stats,
    #     on=["ticker", "start_date", "end_date"]
    # )
    pass