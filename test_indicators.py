import pytest
import pandas as pd
import numpy as np
import indicators as ind
import indicators_v0 as ind_v0  # Assuming your class file is indicators.py
import utils as ut


# @pytest.fixture
# def synthetic_data():
#     """Fixture to create synthetic data."""
#     date_rng = pd.date_range(start='1/1/2020', end='1/02/2020', freq='H')
#     data = pd.DataFrame(date_rng, columns=['date'])
#     data['close'] = np.random.randint(1,100,size=(len(date_rng)))
#     data['symbol'] = 'Test'
#     data.set_index('date', inplace=True)
#     return data


# def test_bollinger_bands_calculations():
#     # Create a simple DataFrame
#     data = pd.DataFrame({
#         'date': pd.date_range(start='1/1/2020', periods=5),
#         'close': [1, 2, 3, 2, 1],
#         'symbol': 'Test'
#     })
#     data.set_index('date', inplace=True)

#     # Initialize BollingerBands object
#     bb = BollingerBands(data, periods=3, n_stdev=1)

#     # Calculate Bollinger Bands
#     bb.get_bb()

#     # The expected results were calculated manually
#     expected_mavg = [np.nan, np.nan, 2.0, 2.3333333333333335, 2.0]
#     expected_uband = [np.nan, np.nan, 3.0, 3.333333333333333, 3.0]
#     expected_lband = [np.nan, np.nan, 1.0, 1.3333333333333333, 1.0]

#     # Check if calculated values match expected results
#     np.testing.assert_allclose(bb.mavg, expected_mavg, rtol=1e-5)
#     np.testing.assert_allclose(bb.uband, expected_uband, rtol=1e-5)
#     np.testing.assert_allclose(bb.lband, expected_lband, rtol=1e-5)



if __name__ == "__main__":

    ticker = "MSFT"
    start_date = "2021-06-01"
    end_date = "2022-09-01"
    df_prices = ut.test_get_data(
        "td", ticker, start_date=start_date, end_date=end_date).sort_index()

    df_prices.to_csv("test_data/msft.csv")

    # # Bollinger bands
    # bb = ind.BollingerBands(df_prices, periods=12, n_stdev=2)
    # bb.get_bb()
    # bb.get_signal()

    # bb.df_bb.to_csv("test_data/bb_new.csv")

    # bbv0 = ind_v0.BollingerBands(n_periods=12, prices=df_prices["close"].to_frame())
    # bbv0.get_dates()
    # bbv0.get_bb()

    # df_bbv0, df_bbv0_ratio = bbv0.indicators_to_df()
    # df_bbv0.to_csv("test_data/bb_old.csv")
    # df_bbv0_ratio.to_csv("test_data/bb_old_ratio.csv")

    # RSI
    # rsi = ind.RSI(df_prices, periods=14)
    # rsi.get_rsi()
    # rsi.get_signal()

    # rsi.df_rsi.to_csv("test_data/rsi_new.csv")

    # rsiv0 = ind_v0.RSI(n_periods=14, prices=df_prices["close"].to_frame())
    # rsiv0.get_rsi()
    # rsiv0.get_dates()

    # df_rsi = rsiv0.indicators_to_df()
    # df_rsi.to_csv("test_data/rsi_old.csv")

    # Stochastic Oscillator

    # stoch = ind.StochasticOscillator(df_prices, periods=14, k_sma_periods=1, d_sma_periods=3)
    # stoch.get_stochastic_oscillator()
    # stoch.get_signal()
    # stoch.df_stoch_oscillator.to_csv("test_data/stoch_osc_new.csv")

    # stoch_old = ind_v0.StochasticOscillator(14, df_prices["close"].to_frame())
    # stoch_old.get_stochastic_oscillator()
    # stoch_old.get_dates()
    # df_stoch_old = stoch_old.indicators_to_df()
    # df_stoch_old.to_csv("test_data/stoch_osc_old.csv")

    aroon = ind.Aroon(df_prices, periods=12)
    aroon.get_aroon()
    aroon.get_signal()
    aroon.df_aroon.to_csv("test_data/aroon_new.csv")

    aroon_old = ind_v0.Aroon(12, df_prices["close"].to_frame())
    aroon_old.get_aroon()
    aroon_old.get_dates()
    df_aroon_down, df_aroon_up, df_aroon_ratio = aroon_old.aroon_to_df()
    df_aroon_down.to_csv("test_data/aroon_down.csv")
    df_aroon_up.to_csv("test_data/aroon_up.csv")
    df_aroon_ratio.to_csv("test_data/aroon_ratio.csv")




