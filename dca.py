import utils as ut
from tda.auth import easy_client
import numpy as np

def create_dca_trades_df(df_prices, symbol, frequency, shares):
    """Generates a trades dataframe for buy and hold strategy

    params:
        df_prices: prices returned from get_data
        symbol: the ticker,
        shares: number of shares bought for buy and hold

    returns:
        df_trades: buy and hold trades dataframe
    """
    df_trades = df_prices.copy()
    for col in df_trades.columns:
        df_trades[col].values[:] = 0

    for i in np.arange(0, len(df_trades)):
        if (i == 0) or (i % frequency == 0):
            df_trades.iloc[i] = shares

    return df_trades

if __name__ == "__main__":
    tick = "QQQ"
    data_source = "td"
    interval = "day"
    start_date = "2021-01-01"
    end_date = "2023-02-01"
    client=ut.get_ameritrade_client()

    df_prices = ut.get_data(
        tick=tick,
        time_interval=interval,
        source=data_source,
        start_date=start_date,
        end_date=end_date,
        period = None,
        client=client)

    frequency = 5
    shares = 10
    print(df_prices.columns.item())
    symbol = "QQQ"

    df_dca = create_dca_trades_df(df_prices, symbol, frequency, shares)

    print(df_dca.head(20))