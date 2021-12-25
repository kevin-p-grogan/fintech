import pandas as pd

from src.data import Munger
from src.financial_model import FinancialModel
from src.portfolio_optimizer import PortfolioOptimizer
from src.results_paths import ResultsPaths
from src.visualizer import Visualizer
from src.config import Config


def main(cfg: Config):
    data = pd.read_pickle(cfg.DATA_FILEPATH)
    munger = Munger(num_days_time_horizon=cfg.NUM_DAYS_TIME_HORIZON, anaysis_column=cfg.ANALYSIS_COLUMN)
    data = munger.preprocess(data)
    portfolio_data = munger.load_portfolio_data(cfg.PORTFOLIO_DATA_FILEPATH)
    paths = ResultsPaths(cfg.RESULTS_BASE_DIR)
    portfolio_data.to_csv(paths.data.current_portfolio)
    financial_model = FinancialModel()
    financial_model.train(data, portfolio_data=portfolio_data, investment_amount=cfg.INVESTMENT_AMOUNT)
    visualizer = Visualizer(paths.plots)
    visualizer.make_financial_model_plots(financial_model, data)
    portfolio_optimizer = PortfolioOptimizer(financial_model, sparsity_importance=cfg.SPARSITY_IMPORTANCE)
    portfolio_optimizer.optimize()
    visualizer.make_portfolio_optimizer_plots(
        portfolio_optimizer,
        data,
        price_limits=cfg.PRICE_LIMITS,
        num_day_in_future=cfg.NUM_DAYS_IN_FUTURE,
        exceedance_probability=cfg.EXCEEDANCE_PROBABILITY
    )
    portfolio_update = portfolio_optimizer.get_portfolio_update(cfg.PORTFOLIO_VOLATILITY)
    visualizer.make_portfolio_update_plot(portfolio_update)
    portfolio_optimizer.save_portfolio_update(paths.data.portfolio_weights, portfolio_update, cfg.METADATA_FILEPATH)


if __name__ == "__main__":
    config = Config.load(__file__)
    main(config)
