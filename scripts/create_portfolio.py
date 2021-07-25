import os

import pandas as pd

from src.data_munger import DataMunger
from src.financial_model import FinancialModel
from src.portfolio_optimizer import PortfolioOptimizer
from src.visualizer import Visualizer

DATA_FILEPATH = "../data/raw/data.pkl"
PORTFOLIO_DATA_FILEPATH = "../data/raw/portfolio.txt"
METADATA_FILE_PATH = "../data/raw/metadata.csv"
RESULTS_FOLDER = "../data/results"
PORTFOLIO_WEIGHTS_FILEPATH = os.path.join(RESULTS_FOLDER, "portfolio_update.csv")
CURRENT_PORTFOLIO_FILEPATH = os.path.join(RESULTS_FOLDER, "current_portfolio.csv")

# Parameters
NUM_DAYS_TIME_HORIZON = 365 / 2.0
ANALYSIS_COLUMN = "Adj Close"
INVESTMENT_AMOUNT = 1000.  # [$]
PORTFOLIO_VOLATILITY = 0.02

# Plotting
price_limits = [0.6, 1.4]


def main():
    data = pd.read_pickle(DATA_FILEPATH)
    data_munger = DataMunger(num_days_time_horizon=NUM_DAYS_TIME_HORIZON, anaysis_column=ANALYSIS_COLUMN)
    data = data_munger.preprocess(data)
    portfolio_data = data_munger.load_portfolio_data(PORTFOLIO_DATA_FILEPATH)
    portfolio_data.to_csv(CURRENT_PORTFOLIO_FILEPATH)
    financial_model = FinancialModel()
    financial_model.train(data, portfolio_data=portfolio_data, investment_amount=INVESTMENT_AMOUNT)
    visualizer = Visualizer(RESULTS_FOLDER)
    visualizer.make_financial_model_plots(financial_model, data)
    portfolio_optimizer = PortfolioOptimizer(financial_model)
    portfolio_optimizer.optimize()
    visualizer.make_portfolio_optimizer_plots(portfolio_optimizer, data, price_limits=price_limits)
    portfolio_optimizer.save_portfolio_update(PORTFOLIO_WEIGHTS_FILEPATH, PORTFOLIO_VOLATILITY, METADATA_FILE_PATH)


if __name__ == "__main__":
    main()
