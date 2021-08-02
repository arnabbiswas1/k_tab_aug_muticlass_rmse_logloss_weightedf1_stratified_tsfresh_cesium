## Observations

### Steps to be executed:

#### Process Raw Data
    $cd /opt/vssexclude/personal/kaggle/kaggle_tab_jan_2021/src
    $python -m scripts.process_raw_data


## Libraries to be installed

conda install pandas numpy matplotlib jupyterlab flake8 seaborn
conda install -c conda-forge pyarrow
conda install -c anaconda lightgbm
conda install -c anaconda scikit-learn
conda install -c conda-forge xgboost 
conda install -c conda-forge catboost
pip install catboost==0.24.4

catboost
xgboost