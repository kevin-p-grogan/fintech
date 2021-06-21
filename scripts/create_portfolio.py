import os

import pandas as pd

from src.data import preprocess, split
from src.financial_model import FinancialModel
from src.portfolio_optimizer import PortfolioOptimizer
from src.visualizer import Visualizer

DATA_FILEPATH = "data/ETF_data.pkl"
METADATA_FILEPATH = "data/ETF_metadata.csv"
RESULTS_FOLDER = "data/results"
RISK_TOLERANCES = [0.0, 1.0, 10.0]
PORTFOLIO_WEIGHTS_FILEPATH = os.path.join(RESULTS_FOLDER, "portfolio_weights.pkl")


def main():
    data = pd.read_pickle(DATA_FILEPATH)
    preprocessed_data = preprocess(data)
    development_data, evaluation_data = split(preprocessed_data)
    financial_model = FinancialModel()
    financial_model.train(development_data)
    visualizer = Visualizer(RESULTS_FOLDER)
    visualizer.make_financial_model_plots(financial_model, development_data)
    portfolio_optimizer = PortfolioOptimizer(financial_model)
    portfolio_optimizer.optimize(evaluation_data, RISK_TOLERANCES)
    portfolio_optimizer.save_portfolio_weights(PORTFOLIO_WEIGHTS_FILEPATH)
    visualizer.make_portfolio_optimizer_plots(portfolio_optimizer, evaluation_data)


if __name__ == "__main__":
    main()
