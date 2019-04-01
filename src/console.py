from src.utils.StockAnalysis import run_analysis


def main():
    ticker = input("Stock Symbol:")
    epochs = input("Number of Epochs (Recommend 100):")
    days = input("Forecast Days (Don't do more than 30...):")
    run_analysis(ticker=ticker.upper(), epochs_num=int(epochs), forecast_days=int(days))


main()
