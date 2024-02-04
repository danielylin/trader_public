import run_strategies as run
import os

if __name__ == "__main__":
    os.chdir("/Users/daniellin/Documents/dan_trades/algo_trader/tables/production")
    tickers = ["IBM", "AAPL", "MSFT", "AAL", "DIS", "SPY", "BAC", "SBUX", "EBAY", "ETSY", "NFLX", "SNAP", "GOOG", "META"]
    indicators = ["bb", "rsi", "stochastic_oscillator", "aroon"]
    historical_export_path = "historical_strategies.csv"
    export_path = "strategies.csv"
    data_source = "td"

    df_latest = run.get_latest_signal(
        tickers, data_source, "basket", indicators).sort_values(by=["current_date", "symbol"], ascending=False)
    run.write(df_latest, historical_export_path, incremental=True)
    run.write(df_latest, export_path, incremental=False)