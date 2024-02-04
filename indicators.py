import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import utils as ut
# import mplfinance as mpf


class BollingerBands(object):
    """A Bollinger Bands object. Bollinger bands are technical indicators that
    use a moving average +/- n std deviations.

    When an equity price moves out of the bounds, this may indicate that it is
    overbought or oversold. We monitor the price and if it comes down below the
    band, then we believe the price will revert back to normal levels.
    """

    def __init__(self, data, periods, n_stdev):
        self.data = data
        self.periods = periods
        self.n_stdev = n_stdev
        self.uband = None
        self.mavg = None
        self.lband = None
        self.df_bb = None
        self.symbol = data["symbol"].head(1).item()

    def get_bb(self):
        """This method returns the upper, middle, and lower bands.

        Args:
            periods (int): number of periods for the moving average
            n_stdev (int): number of standard deviations

        Returns:
            None
        """
        self.uband, self.mavg, self.lband = talib.BBANDS(
            self.data["close"],
            timeperiod=self.periods,
            nbdevup=self.n_stdev,
            nbdevdn=self.n_stdev,
            matype=0)

    def get_signal(self):
        """This method returns the buy or sell signal.
        Returns:
            A series that indicates a bullish or bearish signal for each date.
        """
        if self.uband is None:
            raise ValueError("Upper band is None.")
        if self.lband is None:
            raise ValueError("Lower band is None.")
        if self.mavg is None:
            raise ValueError("Moving avg is None.")

        self.df_bb = pd.concat(
            [
                self.lband,
                self.mavg,
                self.uband,
                self.data["close"]
            ],
            axis=1
        )

        self.df_bb.index = self.data.index

        self.df_bb.columns = [
            "lband",
            "mavg",
            "uband",
            "close"
        ]

        self.df_bb["prev_close"] = (
            self.df_bb["close"].shift(1)
        )

        self.df_bb["prev_uband"] = (
            self.df_bb["uband"].shift(1)
        )
        self.df_bb["prev_lband"] = (
            self.df_bb["lband"].shift(1)
        )

        self.df_bb["is_overbought"] = np.where(
            self.df_bb["close"] > self.df_bb["uband"],
            True, False
        )

        self.df_bb["is_oversold"] = np.where(
            self.df_bb["close"] < self.df_bb["lband"],
            True, False
        )

        self.df_bb["signal"] = np.where(
            (self.df_bb["prev_close"] < self.df_bb["prev_lband"]) &
            (self.df_bb["close"] > self.df_bb["lband"]),
            1,
            np.where(
                (self.df_bb["prev_close"] > self.df_bb["prev_uband"]) &
                (self.df_bb["close"] < self.df_bb["uband"]),
                -1,
                0
            )
        )

    def plot_indicator(self):
        """Plots the Bollinger Bands along with the signals.

        Returns:
            None
        """
        # Retrieve the data from the BollingerBands object

        fig, ax = plt.subplots(figsize=(20, 12))
        # Plot the Bollinger Bands
        ax.plot(self.df_bb.index, self.df_bb["uband"],
                color="#cfe2f3", label="Upper Band")
        ax.plot(self.df_bb.index, self.df_bb["lband"],
                color="#cfe2f3", label="Lower Band")
        ax.plot(self.df_bb.index, self.df_bb["mavg"],
                color="#6c6d70", linestyle="dashed",
                label=f"MA({self.periods})")
        ax.plot(self.df_bb.index, self.df_bb["close"],
                color="#3e3e3f", label="Closing Price")

        # # Plot the signals as dots
        buy_signals = self.df_bb[self.df_bb["signal"] == 1]
        sell_signals = self.df_bb[self.df_bb["signal"] == -1]

        ax.scatter(buy_signals.index, buy_signals["close"],
                   color="#00ab5b", marker="o")
        ax.scatter(sell_signals.index, sell_signals["close"],
                   color="#ee1d23", marker="o")

        # Annotate buy signals
        for i in range(len(buy_signals)):
            ax.annotate(round(buy_signals["close"].iloc[i], 2),
                        (buy_signals.index[i], buy_signals["close"].iloc[i]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=8,
                        color="black")

        # Annotate sell signals
        for i in range(len(sell_signals)):
            ax.annotate(round(sell_signals["close"].iloc[i], 2),
                        (sell_signals.index[i], sell_signals["close"].iloc[i]),
                        textcoords="offset points",
                        xytext=(0, -15),
                        ha="center",
                        fontsize=8,
                        color="black")

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        plt.legend()
        ax.set_title(f"Bollinger Bands: {self.symbol}")
        plt.grid(True)
        plt.show()


class RSI(object):
    """
    A Relative Strength Index object. RSI is a momentum indicator that measures
    the speed and the magnitude of a security"s recent price chances.

    The RSI is displayed as an oscillator on a line graph from 0 to 100.
    """

    def __init__(self, data, periods=14):
        self.data = data
        self.periods = periods
        self.rsi = None
        self.df_rsi = None
        self.symbol = data["symbol"].head(1).item()

    def get_rsi(self):
        """This function calculates RSI."""
        self.rsi = talib.RSI(self.data["close"], timeperiod=self.periods)

    def get_signal(self, ulevel=70, llevel=30):
        """This method fetches the trading signal.

        Args:
            ulevel: the RSI level that indicates overbought
            llevel: the RSI level that indicates oversold
        """
        self.df_rsi = pd.concat(
            [
                self.rsi,
                self.data["close"]
            ],
            axis=1
        )

        self.df_rsi.index = self.data.index

        self.df_rsi.columns = ["rsi", "close"]

        self.df_rsi["is_overbought"] = np.where(
            self.df_rsi["rsi"] > ulevel,
            True, False
        )

        self.df_rsi["is_oversold"] = np.where(
            self.df_rsi["close"] < llevel,
            True, False
        )

        self.df_rsi["ulevel"] = ulevel
        self.df_rsi["llevel"] = llevel

        self.df_rsi["prev_rsi"] = (
            self.df_rsi["rsi"].shift(1)
        )

        self.df_rsi["signal"] = np.where(
            (self.df_rsi["prev_rsi"] < self.df_rsi["llevel"]) &
            (self.df_rsi["rsi"] > self.df_rsi["llevel"]),
            1,
            np.where(
                (self.df_rsi["prev_rsi"] > self.df_rsi["ulevel"]) &
                (self.df_rsi["rsi"] < self.df_rsi["ulevel"]),
                -1,
                0
            )
        )

    def plot_indicator(self):
        """This method plots the RSI indicator below the equity price.
        """

        fig = plt.figure(figsize=(20, 12))

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(self.df_rsi.index, self.df_rsi["close"],
                 color="#3e3e3f", label="Closing Price")

        ax1.set_xlim([
            pd.to_datetime(self.df_rsi.index.min()),
            pd.to_datetime(self.df_rsi.index.max())])

        ax1.set_title(f"Equity Price: {self.symbol}")

        ax2 = fig.add_subplot(2, 1, 2)

        ax2.plot(self.df_rsi.index, self.df_rsi["ulevel"],
                 color="#cfe2f3", linestyle="dashed")
        ax2.plot(self.df_rsi.index, self.df_rsi["llevel"],
                 color="#cfe2f3", linestyle="dashed")
        ax2.plot(self.df_rsi.index, self.df_rsi["rsi"], color="#6c6d70")

        ax2.set_title(f"RSI: {self.symbol}")

        buy_signals = self.df_rsi[self.df_rsi["signal"] == 1]
        sell_signals = self.df_rsi[self.df_rsi["signal"] == -1]

        ax1.scatter(buy_signals.index, buy_signals["close"],
                    color="#00ab5b", marker="o")
        ax1.scatter(sell_signals.index, sell_signals["close"],
                    color="#ee1d23", marker="o")

        # Annotate buy signals
        for i in range(len(buy_signals)):
            ax1.annotate(round(buy_signals["close"].iloc[i], 2),
                         (buy_signals.index[i], buy_signals["close"].iloc[i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha="center",
                         fontsize=8,
                         color="black")

        # Annotate sell signals
        for i in range(len(sell_signals)):
            ax1.annotate(round(sell_signals["close"].iloc[i], 2),
                         (sell_signals.index[i], sell_signals["close"].iloc[i]),
                         textcoords="offset points",
                         xytext=(0, -15),
                         ha="center",
                         fontsize=8,
                         color="black")

        ax2.scatter(buy_signals.index, buy_signals["rsi"],
                    color="#00ab5b", marker="o")
        ax2.scatter(sell_signals.index, sell_signals["rsi"],
                    color="#ee1d23", marker="o")
        ax2.set_xlim([
            pd.to_datetime(self.df_rsi.index.min()),
            pd.to_datetime(self.df_rsi.index.max())])

        plt.tight_layout()
        plt.show()


class MACD(object):
    """
    A Moving average convergence divergence object. MACD  is a trend-following
    momentum indicator that shows the relationship between two exponential
    moving averages (EMAs) of a security"s price.
    """

    def __init__(self, data, periods=14):
        self.data = data
        self.periods = periods
        self.macd = None
        self.symbol = data["symbol"].head(1).item()

    def get_macd(self):
        """This function calculates MACD."""
        self.macd, self.macdsignal, self.macdhist = talib.MACD(
            self.data["close"], fastperiod=12, slowperiod=26, signalperiod=9)

    def get_signal(self):
        """This method fetches the trading signal.
        """
        self.df_macd = pd.concat(
            [
                self.macd,
                self.macdsignal,
                self.data["close"]
            ],
            axis=1
        )

        self.df_macd.index = self.data.index

        self.df_macd.columns = ["macd", "macdsignal", "close"]

        self.df_macd["prev_macd"] = (
            self.df_macd["macd"].shift(1)
        )
        self.df_macd["prev_macdsignal"] = (
            self.df_macd["macdsignal"].shift(1)
        )

        self.df_macd["signal"] = np.where(
            (self.df_macd["prev_macd"] < self.df_macd["prev_macdsignal"]) &
            (self.df_macd["macd"] > self.df_macd["macdsignal"]),
            1,
            np.where(
                (self.df_macd["prev_macd"] > self.df_macd["prev_macdsignal"]) &
                (self.df_macd["macd"] < self.df_macd["macdsignal"]), -1, 0
            )
        )

    def plot_indicator(self):
        """This method plots the MACD indicators below the equity price.
        """

        fig = plt.figure(figsize=(20, 12))

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(self.df_macd.index, self.df_macd["close"],
                 color="#3e3e3f", label="Closing Price")

        ax1.set_title(f"Equity Price: {self.symbol}")

        ax1.set_xlim([
            pd.to_datetime(self.df_macd.index.min()),
            pd.to_datetime(self.df_macd.index.max())])

        ax2 = fig.add_subplot(2, 1, 2)

        ax2.plot(self.df_macd.index, self.df_macd["macdsignal"],
                 color="#cfe2f3", linestyle="dashed")
        ax2.plot(self.df_macd.index, self.df_macd["macd"], color="#6c6d70")

        ax2.set_title(f"MACD: {self.symbol}")

        buy_signals = self.df_macd[self.df_macd["signal"] == 1]
        sell_signals = self.df_macd[self.df_macd["signal"] == -1]

        ax1.scatter(buy_signals.index, buy_signals["close"],
                    color="#00ab5b", marker="o")
        ax1.scatter(sell_signals.index, sell_signals["close"],
                    color="#ee1d23", marker="o")

        # Annotate buy signals
        for i in range(len(buy_signals)):
            ax1.annotate(round(buy_signals["close"].iloc[i], 2),
                         (buy_signals.index[i], buy_signals["close"].iloc[i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha="center",
                         fontsize=8,
                         color="black")

        # Annotate sell signals
        for i in range(len(sell_signals)):
            ax1.annotate(round(sell_signals["close"].iloc[i], 2),
                         (sell_signals.index[i], sell_signals["close"].iloc[i]),
                         textcoords="offset points",
                         xytext=(0, -15),
                         ha="center",
                         fontsize=8,
                         color="black")

        ax2.scatter(buy_signals.index, buy_signals["macd"],
                    color="#00ab5b", marker="o")
        ax2.scatter(sell_signals.index, sell_signals["macd"],
                    color="#ee1d23", marker="o")

        ax2.set_xlim([
            pd.to_datetime(self.df_macd.index.min()),
            pd.to_datetime(self.df_macd.index.max())])

        plt.tight_layout()
        plt.show()


class Aroon(object):
    """
    An Aroon object. The Aroon indicator is a technical indicator that is used
    to identify trend changes in the price of an asset, as well as the strength
    of that trend. In essence, the indicator measures the time between highs
    and the time between lows over a time period.

    The Aroon up and Aroon down indicators fluctuate between 0 and 100.
    The indicator focuses on the last 25 periods, but is scaled to 0 and 100.

    An Aroon Up reading above 50 means the price made a new high
    within the last 12.5 periods. A reading near 100 means a high was seen
    very recently.

    The same concepts apply to the Down Aroon. When it is above 50, a low was
    witnessed within the 12.5 periods. A Down reading near 100 means a low was
    seen very recently.
    """

    def __init__(self, data, periods=25):
        self.data = data
        self.periods = periods
        self.aroondown = None
        self.aroonup = None
        self.symbol = data["symbol"].head(1).item()

    def get_aroon(self):
        """This function calculates Aroon."""
        self.aroondown, self.aroonup = talib.AROON(
            self.data["high"], self.data["low"], timeperiod=self.periods)

    def get_signal(self):
        """This method fetches the trading signal.
        """
        self.df_aroon = pd.concat(
            [
                self.aroondown,
                self.aroonup,
                self.data["close"]
            ],
            axis=1
        )

        self.df_aroon.index = self.data.index

        self.df_aroon.columns = ["aroondown", "aroonup", "close"]

        self.df_aroon["prev_aroondown"] = (
            self.df_aroon["aroondown"].shift(1)
        )

        self.df_aroon["prev_aroonup"] = (
            self.df_aroon["aroonup"].shift(1)
        )

        self.df_aroon["signal"] = np.where(
            (self.df_aroon["prev_aroonup"] < self.df_aroon["prev_aroondown"]) &
            (self.df_aroon["aroonup"] > self.df_aroon["aroondown"]),
            1,
            np.where(
                (self.df_aroon["prev_aroonup"] > self.df_aroon["prev_aroondown"]) &
                (self.df_aroon["aroonup"] < self.df_aroon["aroondown"]), -1, 0
            )
        )

    def plot_indicator(self):
        """This method plots the RSI indicator below the equity price.
        """

        fig = plt.figure(figsize=(20, 12))

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(self.df_aroon.index, self.df_aroon["close"],
                 color="#3e3e3f", label="Closing Price")

        ax1.set_title(f"Equity Price: {self.symbol}")

        ax1.set_xlim([
            pd.to_datetime(self.df_aroon.index.min()),
            pd.to_datetime(self.df_aroon.index.max())])

        ax2 = fig.add_subplot(2, 1, 2)

        ax2.plot(self.df_aroon.index, self.df_aroon["aroonup"],
                 color="#cfe2f3")
        ax2.plot(self.df_aroon.index, self.df_aroon["aroondown"],
                 color="#6c6d70")

        ax2.set_title(f"Aroon: {self.symbol}")

        buy_signals = self.df_aroon[self.df_aroon["signal"] == 1]
        sell_signals = self.df_aroon[self.df_aroon["signal"] == -1]

        ax1.scatter(buy_signals.index, buy_signals["close"],
                    color="#00ab5b", marker="o")
        ax1.scatter(sell_signals.index, sell_signals["close"],
                    color="#ee1d23", marker="o")

        # Annotate buy signals
        for i in range(len(buy_signals)):
            ax1.annotate(round(buy_signals["close"].iloc[i], 2),
                         (buy_signals.index[i], buy_signals["close"].iloc[i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha="center",
                         fontsize=8,
                         color="black")

        # Annotate sell signals
        for i in range(len(sell_signals)):
            ax1.annotate(round(sell_signals["close"].iloc[i], 2),
                         (sell_signals.index[i], sell_signals["close"].iloc[i]),
                         textcoords="offset points",
                         xytext=(0, -15),
                         ha="center",
                         fontsize=8,
                         color="black")

        ax2.scatter(buy_signals.index, buy_signals["aroonup"],
                    color="#00ab5b", marker="o")
        ax2.scatter(sell_signals.index, sell_signals["aroondown"],
                    color="#ee1d23", marker="o")
        ax2.set_xlim([
            pd.to_datetime(self.df_aroon.index.min()),
            pd.to_datetime(self.df_aroon.index.max())])

        plt.tight_layout()
        plt.show()


class StochasticOscillator(object):
    """
    A Stochastic Oscillator object. The Stochastic Oscillator is a technical
    indicator that compares the closing price of an asset to its price range
    over a given time period. It is used to identify overbought and oversold
    conditions and potential trend reversals.

    The Stochastic Oscillator fluctuates between 0 and 100.
    Readings above 80 are considered overbought, and readings below 20 are
    considered oversold.

    The indicator consists of two lines: %K and %D. %K is the faster line,
    while %D is a moving average of %K.

    The standard settings for the Stochastic Oscillator are a 14-period %K
    and a 3-period simple moving average for %D.
    """

    def __init__(self, data, periods=14, k_sma_periods=1, d_sma_periods=3, closing_prices=True):
        self.data = data
        self.periods = periods
        self.k_sma_periods = k_sma_periods
        self.d_sma_periods = d_sma_periods
        self.k_values = None
        self.d_values = None
        self.symbol = data["symbol"].head(1).item()
        self.closing_prices = closing_prices

    def get_stochastic_oscillator(self):
        """This function calculates the Stochastic Oscillator."""

        if self.closing_prices:
            self.k_values, self.d_values = talib.STOCH(
                self.data["close"], self.data["close"], self.data["close"],
                fastk_period=self.periods, slowk_period=self.k_sma_periods,
                slowd_period=self.d_sma_periods)
        else:
            self.k_values, self.d_values = talib.STOCH(
                self.data["high"], self.data["low"], self.data["close"],
                fastk_period=self.periods, slowk_period=self.k_sma_periods,
                slowd_period=self.d_sma_periods)

    def get_signal(self, overbought_threshold=80, oversold_threshold=20):
        """This method fetches the trading signal based on overbought and oversold conditions."""
        self.df_stoch_oscillator = pd.concat(
            [
                self.k_values,
                self.d_values,
                self.data["close"]
            ],
            axis=1
        )

        self.df_stoch_oscillator.index = self.data.index
        self.df_stoch_oscillator.columns = ["%K", "%D", "close"]

        self.df_stoch_oscillator["%prev_K"] = self.df_stoch_oscillator["%K"].shift(1)

        self.df_stoch_oscillator["signal"] = np.where(
            (self.df_stoch_oscillator["%prev_K"] < oversold_threshold) &
            (self.df_stoch_oscillator["%K"] > oversold_threshold),
            1,
            np.where(
                (self.df_stoch_oscillator["%prev_K"] > overbought_threshold) &
                (self.df_stoch_oscillator["%K"] < overbought_threshold),
                -1,
                0
            )
        )

    def plot_indicator(self):
        """This method plots the Stochastic Oscillator below the equity price."""
        fig = plt.figure(figsize=(20, 12))

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(self.df_stoch_oscillator.index, self.df_stoch_oscillator["close"],
                 color="#3e3e3f", label="Closing Price")

        ax1.set_title(f"Equity Price: {self.symbol}")
        ax1.set_xlim([
            pd.to_datetime(self.df_stoch_oscillator.index.min()),
            pd.to_datetime(self.df_stoch_oscillator.index.max())])

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(self.df_stoch_oscillator.index, self.df_stoch_oscillator["%K"],
                 color="#cfe2f3", label="%K")
        ax2.plot(self.df_stoch_oscillator.index, self.df_stoch_oscillator["%D"],
                 color="#6c6d70", label="%D")

        ax2.axhline(80, color="red", linestyle="--", linewidth=1)
        ax2.axhline(20, color="green", linestyle="--", linewidth=1)

        buy_signals = self.df_stoch_oscillator[self.df_stoch_oscillator["signal"] == 1]
        sell_signals = self.df_stoch_oscillator[self.df_stoch_oscillator["signal"] == -1]

        ax1.scatter(buy_signals.index, buy_signals["close"],
                    color="#00ab5b", marker="o")
        ax1.scatter(sell_signals.index, sell_signals["close"],
                    color="#ee1d23", marker="o")

        # Annotate buy signals
        for i in range(len(buy_signals)):
            ax1.annotate(round(buy_signals["close"].iloc[i], 2),
                         (buy_signals.index[i], buy_signals["close"].iloc[i]),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha="center",
                         fontsize=8,
                         color="black")

        # Annotate sell signals
        for i in range(len(sell_signals)):
            ax1.annotate(round(sell_signals["close"].iloc[i], 2),
                         (sell_signals.index[i], sell_signals["close"].iloc[i]),
                         textcoords="offset points",
                         xytext=(0, -15),
                         ha="center",
                         fontsize=8,
                         color="black")

        ax2.scatter(buy_signals.index, buy_signals["%K"],
                    color="#00ab5b", marker="o")
        ax2.scatter(sell_signals.index, sell_signals["%K"],
                    color="#ee1d23", marker="o")
        ax2.set_title(f"Stochastic Oscillator: {self.symbol}")

        ax1.legend()
        ax2.legend()
        plt.tight_layout()
        plt.show()


def test_indicators():
    df = ut.test_get_data("td", "MSFT", period="6mo")
    # mpf.plot(df,type="candle",mav=(3,6,9),volume=True,show_nontrading=True)

    # bb = BollingerBands(df, periods=20, n_stdev=2)
    # bb.get_bb()
    # bb.get_signal()
    # bb.plot_indicator()
    # plt.close()

    # rsi = RSI(df, periods=14)
    # rsi.get_rsi()
    # rsi.get_signal()
    # rsi.plot_indicator()

    # macd = MACD(df, periods=14)
    # macd.get_macd()
    # macd.get_signal()
    # macd.plot_indicator()

    # aroon = Aroon(df, periods=25)
    # aroon.get_aroon()
    # aroon.get_signal()
    # aroon.plot_indicator()

    stoch_osc = StochasticOscillator(df)
    stoch_osc.get_stochastic_oscillator()
    stoch_osc.get_signal()
    stoch_osc.plot_indicator()


if __name__ == "__main__":
    test_indicators()
