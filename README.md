# fintech
Seeks to understand possible investments and attempts to build a diversified portfolio.

## Requirements
All requirements are python dependencies given in *requirements.txt*. One can install these via

`pip install -r requirements.txt`

## Use
*scripts/fetch_data.py* fetches price information on assets of interest. 
Information on the assets of interest is given in the *data/raw/metadata.csv*
*scripts/create_portfolio.py* is run to perform analytics on the assets of interest and suggest a new investments.
Current portfolio data can be inputed and taken into account via the *data/raw/portfolio.txt* file. An example is given.
A *config.yaml* file is required for the run configuration parameters.

