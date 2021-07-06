import os

import pandas as pd
import numpy as np

from src.data_munger import DataMunger
from src.financial_model import FinancialModel
from src.portfolio_optimizer import PortfolioOptimizer
from src.visualizer import Visualizer

DATA_FILEPATH = "../data/raw/ETF_data.pkl"
METADATA_FILEPATH = "../data/raw/ETF_metadata.csv"
RESULTS_FOLDER = "../data/results"
RISKS = list(np.logspace(-4, -2, 10))
PORTFOLIO_WEIGHTS_FILEPATH = os.path.join(RESULTS_FOLDER, "portfolio_weights.pkl")

# Parameters
NUM_DAYS_TIME_HORIZON = 365
ANALYSIS_COLUMN = "Adj Close"
NUM_DAYS_PREDICTION_PERIOD = 30


def main():
    data = pd.read_pickle(DATA_FILEPATH)
    data_munger = DataMunger(num_days_time_horizon=NUM_DAYS_TIME_HORIZON, anaysis_column=ANALYSIS_COLUMN)
    data = data_munger.preprocess(data)
    financial_model = FinancialModel(num_days_prediction_period=NUM_DAYS_PREDICTION_PERIOD)
    financial_model.train(data)
    visualizer = Visualizer(RESULTS_FOLDER)
    visualizer.make_financial_model_plots(financial_model, data)
    portfolio_optimizer = PortfolioOptimizer(financial_model)
    portfolio_optimizer.optimize()
    portfolio_optimizer.save_portfolio_weights(PORTFOLIO_WEIGHTS_FILEPATH)
    visualizer.make_portfolio_optimizer_plots(portfolio_optimizer, data, RISKS)


if __name__ == "__main__":
    main()
