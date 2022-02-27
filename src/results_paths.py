import os
from collections import namedtuple
from datetime import datetime


class ResultsPaths:
    _DataFilePaths: namedtuple = namedtuple("DataFilePaths", ["portfolio_weights", "current_portfolio", "config"])

    def __init__(self,
                 base_dir: str,
                 current_portfolio_filename: str = "current_portfolio.csv",
                 portfolio_weights_filename: str = "portfolio_weights.csv",
                 config_filename: str = "config.yaml",):
        results_directory = self._create_results_directory(base_dir)
        self.data = ResultsPaths._DataFilePaths(
            current_portfolio=os.path.join(results_directory, current_portfolio_filename),
            portfolio_weights=os.path.join(results_directory, portfolio_weights_filename),
            config=os.path.join(results_directory, config_filename),
        )
        self.plots = self._create_plots_directory(results_directory)

    @staticmethod
    def _create_results_directory(base_dir: str) -> str:
        current_time = datetime.now().isoformat()
        results_dir_name = f"run-{current_time}"
        results_dir_path = os.path.join(base_dir, results_dir_name)
        os.mkdir(results_dir_path)
        return results_dir_path

    @staticmethod
    def _create_plots_directory(results_directory: str) -> str:
        plots_dir_path = os.path.join(results_directory, "plots")
        os.mkdir(plots_dir_path)
        return plots_dir_path
