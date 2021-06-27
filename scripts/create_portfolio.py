import os

import pandas as pd

from src.data_munger import DataMunger
from src.financial_model import FinancialModel
from src.portfolio_optimizer import PortfolioOptimizer
from src.visualizer import Visualizer

DATA_FILEPATH = "data/ETF_data.pkl"
METADATA_FILEPATH = "data/ETF_metadata.csv"
RESULTS_FOLDER = "data/results"
RISK_TOLERANCES = [0.1, 1.0, 10.0]
PORTFOLIO_WEIGHTS_FILEPATH = os.path.join(RESULTS_FOLDER, "portfolio_weights.pkl")

# Parameters
NUM_DAYS_TIME_HORIZON = 365
ANALYSIS_COLUMN = "Adj Close"

def main():
    data = pd.read_pickle(DATA_FILEPATH)
    data_munger = DataMunger(num_days_time_horizon=NUM_DAYS_TIME_HORIZON, anaysis_column=ANALYSIS_COLUMN)
    data = data_munger.preprocess(data)
    financial_model = FinancialModel()
    financial_model.train(data)
    visualizer = Visualizer(RESULTS_FOLDER)
    visualizer.make_financial_model_plots(financial_model, data)
    portfolio_optimizer = PortfolioOptimizer(financial_model)
    portfolio_optimizer.optimize(RISK_TOLERANCES)
    portfolio_optimizer.save_portfolio_weights(PORTFOLIO_WEIGHTS_FILEPATH)
    visualizer.make_portfolio_optimizer_plots(portfolio_optimizer, data)


if __name__ == "__main__":
    main()
