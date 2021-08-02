Repository for **Kaggle TPS July 2021 - Air Quality - Time Series - RMSLE - Multiple Targets**

## Steps to execute:

1. Clone the source code from github under <PROJECT_HOME> directory.
> git clone https://github.com/arnabbiswas1/kaggle_tab_jun.git

This will create the following directory structure:
> <PROJECT_HOME>/kaggle_tab_jun

2. Create conda env:
> conda env create --file environment.yml

3. Go to the raw data directory at `<PROJECT_HOME>/kaggle_tab_jun/data/raw`. Download dataset from Kaggle:
> kaggle competitions download -c tabular-playground-series-jun-2021

4. Unzip the data:
> unzip tabular-playground-series-jun-2021.zip

5. Set the value of variable `HOME_DIR` at `<PROJECT_HOME>/kaggle_tab_jun/src/config/constants.py` with the absolute path of `<PROJECT_HOME>/kaggle_tab_jun`

6. To process raw data into parquet format, go to `<PROJECT_HOME>/kaggle_tab_jun/src`. Execute the following:
> python -m scripts.process_raw_data

7. To trigger feature engineering, go to `<PROJECT_HOME>/kaggle_tab_jun/src`. Execute the following:
> python -m scripts.create_fetaures


## Note:

Following is needed for visualizing plots for optuna using plotly (i.e. plotly dependency):

> jupyter labextension install jupyterlab-plotly@4.14.3
