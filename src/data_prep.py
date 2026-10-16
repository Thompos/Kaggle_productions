import math as ma
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


def prepare_data(kag_train, kag_test):

    kag_train_cl = kag_train.copy()
    kag_test_dat_cl = kag_test.copy()

    col2ch = 'MasVnrArea'
    kag_train_cl.loc[:, col2ch ] = kag_train_cl[col2ch].fillna(0)
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna(0)

    ## must do the imputations
    # RF
    train_all = kag_train_cl.copy()
    test_all = kag_test_dat_cl.copy()

    X = pd.concat([train_all, test_all], ignore_index=True)

    exclude = ['ID', 'SalePrice', 'LotFrontage'] # 

    features = [
    col for col in X.columns
    if col not in exclude
    and X[col].notna().all()
    ]

    X = X[features + ['LotFrontage']].copy()


    for col in X.select_dtypes(include=['object', 'str']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])


    train_all  = X.iloc[:len(train_all)]
    test_all  = X.iloc[len(train_all):]


    train_rf = train_all.dropna(subset=['LotFrontage']).copy()
    test_rf = test_all.dropna(subset=['LotFrontage']).copy()


    X_train_rf = train_rf[features].copy() # train_rf[features].copy()
    y_train_rf = train_rf['LotFrontage']
    X_test_rf = test_rf[features].copy()
    y_test_rf = test_rf['LotFrontage']


    rf_reg = RandomForestRegressor(random_state=69, n_jobs=-1)

    param_grid = {
    'n_estimators': [100, 250, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    }


    grid_search = GridSearchCV(
    estimator=rf_reg,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    refit=True
    )

    grid_search.fit(X_train_rf, y_train_rf)

    best_model = grid_search.best_estimator_

    y_test_rf_preds = best_model.predict(X_test_rf)

    final_mse = mean_squared_error(y_test_rf, y_test_rf_preds)
    final_r2 = r2_score(y_test_rf, y_test_rf_preds)

    print("RF Lotfrontage imputation Held-out MSE:", final_mse)
    print("RF Lotfrontage Held-out R²:", final_r2)
    print("RF Lotfrontage Best parameters:", grid_search.best_params_)

    train_impute_cases = train_all[train_all['LotFrontage'].isna()].copy()
    test_impute_cases = test_all[test_all['LotFrontage'].isna()].copy()

    X_trp = train_impute_cases[features].copy()
    X_tsp = test_impute_cases[features].copy()

    y_train_impute = best_model.predict(X_trp)
    y_test_impute = best_model.predict(X_tsp)


    kag_train_cl.loc[kag_train_cl['LotFrontage'].isna(), 'LotFrontage'] = y_train_impute
    kag_test_dat_cl.loc[kag_test_dat_cl['LotFrontage'].isna(), 'LotFrontage'] = y_test_impute

    ###
    col2ch = 'SaleType'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('WD')

    col2ch = 'MiscFeature'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('RL')

    col2ch = 'MiscFeature'
    kag_train_cl.loc[:,col2ch ] = kag_train_cl[col2ch].fillna('None')
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('None')

    col2ch = 'Alley'
    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna('None')
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('None')

    col2ch = 'FireplaceQu'
    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna('None')
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('None')

    col2ch = 'PoolQC'

    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna(
    kag_train_cl['PoolArea'].apply(lambda x: 'None' if x == 0 else 'Gd') )
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna(
    kag_test_dat_cl['PoolArea'].apply(lambda x: 'None' if x == 0 else 'Gd') )

    col2ch = 'MasVnrType'
    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna('Render')
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('Render')

    col2ch = 'BsmtQual'
    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna('None')
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('None')

    col2ch = 'GarageCond'

    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna(
    kag_train_cl['GarageType'].apply(lambda x: 'TA' if x == 'Detchd' else 'None') )
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna(
    kag_test_dat_cl['GarageType'].apply(lambda x: 'TA' if x == 'Detchd' else 'None') )

    col2ch = 'GarageQual'
    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna('None')
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('None')

    col2ch = 'GarageFinish'
    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna('None')
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('None')

    col2ch = 'GarageType'
    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna('None')
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('None')

    col2ch = 'GarageArea'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna(0)

    col2ch = 'GarageCars'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna(0)

    col2ch = 'GarageYrBlt'
    mask = kag_train_cl[col2ch].isna()
    kag_train_cl.loc[mask, col2ch] = kag_train_cl.loc[mask, 'YearBuilt']

    mask = kag_test_dat_cl[col2ch].isna()
    kag_test_dat_cl.loc[mask, col2ch] = kag_test_dat_cl.loc[mask, 'YearBuilt']

    col2ch = 'Electrical'
    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna('SBrkr')

    col2ch = 'BsmtFinType2'
    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna('None')
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('None')

    col2ch = 'BsmtFinType1'
    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna('None')
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('None')

    col2ch = 'BsmtExposure'
    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna('None')
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('None')

    col2ch = 'BsmtCond'
    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna('None')
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('None')

    col2ch = 'BsmtQual'
    kag_train_cl.loc[:, col2ch] = kag_train_cl[col2ch].fillna('None')
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('None')

    col2ch = 'Fence'
    kag_train_cl.drop(columns = [col2ch], inplace = True)
    kag_test_dat_cl.drop(columns = [col2ch], inplace = True)

    col2ch = 'Functional'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('Typ')

    col2ch = 'KitchenQual'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('TA')

    col2ch = 'BsmtHalfBath'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna(0)

    col2ch = 'BsmtFullBath'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna(0)

    col2ch = 'TotalBsmtSF'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna(0)

    col2ch = 'BsmtUnfSF'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna(0)

    col2ch = 'BsmtFinSF1'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna(0)

    col2ch = 'BsmtFinSF2'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna(0)

    col2ch = 'Exterior1st'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('VinylSd')

    col2ch = 'Exterior2nd'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('VinylSd')

    col2ch = 'Utilities'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('AllPub')

    col2ch = 'MSZoning'
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna('RL')

    return kag_train_cl, kag_test_dat_cl


#kag_train.to_pickle("orig_train1.pkl")
#kag_test.to_pickle("orig_test1.pkl")
