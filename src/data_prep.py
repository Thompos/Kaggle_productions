import math as ma
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
# import string

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = Path(__file__).resolve().parent.parent/"models"/"imputation"/"lotfrontage_rf.joblib"


def prepare_data1(kag_train, kag_test):

    kag_train_cl = kag_train.copy()
    kag_test_dat_cl = kag_test.copy()

    col2ch = 'MasVnrArea'
    kag_train_cl.loc[:, col2ch ] = kag_train_cl[col2ch].fillna(0)
    kag_test_dat_cl.loc[:, col2ch] = kag_test_dat_cl[col2ch].fillna(0)

    ## must do the imputations
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

    if MODEL_PATH.exists():
        best_model = joblib.load(MODEL_PATH)
        print("Loaded saved LotFrontage Random Forest model.")
    else:
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

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

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



def prepare_data2(clean_train, clean_test):

    mask = clean_train['YearRemodAdd'] == 1950
    clean_train.loc[mask, 'YearRemodAdd'] = clean_train.loc[mask, 'YearBuilt']

    mask = clean_test['YearRemodAdd'] == 1950
    clean_test.loc[mask, 'YearRemodAdd'] = clean_test.loc[mask, 'YearBuilt']


    clean_train['Source'] = 'train'
    clean_test['Source'] = 'test'
    clean_test['SalePrice'] = np.nan

    clean_ALL =  pd.concat([clean_train, clean_test], ignore_index=True)


    clean_ALL = pd.get_dummies(
        clean_ALL,
        columns=['MSSubClass'],
        prefix='MSSubClass',
        drop_first=False,
        dtype=int
    )

    var = 'MSZoning'
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)


    var = 'Street'
    clean_ALL[var] = clean_ALL[var].apply(lambda x: 0 if x == 'Grvl' else 1)


    var = 'Alley'
    var_map = { 'None': 0, 'Grvl':1, 'Pave': 2 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'LotShape'
    var_map = { 'Reg': 0, 'IR1':1 , 'IR2':2, 'IR3': 3 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'LandContour'
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)


    var = 'Utilities'
    clean_ALL.drop([var], inplace = True, axis = 1)


    var = 'LotConfig'
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)


    var = 'LandSlope'
    var_map = { 'Gtl':0, 'Mod':1, 'Sev':2}
    clean_ALL[var] = clean_ALL[var].map(var_map)


    clean_ALL['Traffic'] = clean_ALL.apply( 
                    lambda row: 2 if row['Condition1'] == 'Artery' or row['Condition2'] == 'Artery' else 
                    (1 if row['Condition1'] == 'Feedr' or row['Condition2'] == 'Feedr' else 0), axis=1 )

    clean_ALL['Positive'] = [  2 if (x == 'PosA' or y == 'PosA') else 
                            (1 if (x == 'PosN' or y == 'PosN') else 0) 
                            for x, y in zip(clean_ALL['Condition1'], clean_ALL['Condition2']) ]

    rail_map = {
        'RRAe': 4,
        'RRAn': 3,
        'RRNe': 2,
        'RRNn': 1 }

    temp_c1 = clean_ALL['Condition1'].map(rail_map).fillna(0).astype(int)
    temp_c2 = clean_ALL['Condition2'].map(rail_map).fillna(0).astype(int)
    clean_ALL['Rail'] = [max(x,y) for x,y in zip(temp_c1, temp_c2)]
    clean_ALL.drop(columns=['Condition1', 'Condition2'], inplace=True)


    var = 'BldgType'
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)


    var = 'HouseStyle'
    var_map = { 'SLvl': -3,
                'SFoyer': -2,
                '1Story': 0,
                '1.5Fin': 1, '1.5Unf': 1,
                '2Story': 2,
                '2.5Unf': 3, '2.5Fin': 3   }
    clean_ALL['Stories'] = clean_ALL[var].map(var_map)


    var_map = { 'SLvl': 1, 'SFoyer': 1, '1Story': 1, '1.5Fin': 1, '2Story': 1, '2.5Fin': 1,
            '1.5Unf': 0, '2.5Unf': 0   }
    clean_ALL['Finished'] = clean_ALL[var].map(var_map) 


    clean_ALL.drop(['HouseStyle'], inplace = True, axis = 1)


    var = 'RoofStyle'
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)


    var = 'RoofMatl'
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)


    clean_ALL['Exterior'] = [x +'_'+ y for x,y in zip(clean_ALL['Exterior1st'], clean_ALL['Exterior2nd'])]
    clean_ALL.drop(['Exterior1st'], inplace = True, axis = 1)
    clean_ALL.drop(['Exterior2nd'], inplace = True, axis = 1)


    var = 'MasVnrType'
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)


    var = 'ExterQual'
    var_map = { 'Fa':0, 'TA':1, 'Gd':2, 'Ex':3 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'ExterCond'
    var_map = { 'Po':0, 'Fa':1, 'TA':2, 'Gd': 3, 'Ex':4 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'Foundation'
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)


    mask = (clean_ALL['BsmtQual'] != 'None') & (clean_ALL['BsmtCond'] == 'None')
    clean_ALL.loc[mask, 'BsmtCond'] = 'Unf'


    mask = (clean_ALL['BsmtQual'] == 'None') & (clean_ALL['BsmtCond'] != 'None')
    clean_ALL.loc[mask, 'BsmtQual'] = 'Unf'


    mask = (clean_ALL['BsmtQual'] != 'None') & (clean_ALL['BsmtExposure'] == 'None')
    clean_ALL.loc[mask, 'BsmtExposure'] = 'No'


    mask = (clean_ALL['BsmtQual'] != 'None') & (clean_ALL['BsmtFinType2'] == 'None')
    clean_ALL.loc[mask, 'BsmtFinType2'] = 'Unf'


    var = 'BsmtQual'
    var_map =  { 'None': -1, 'Unf':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd': 4, 'Ex':5 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'BsmtCond'
    var_map =  { 'None': -1, 'Unf':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd': 4, 'Ex':5 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'BsmtExposure'
    var_map =  { 'None': -1, 'No':1, 'Mn':2, 'Av':3, 'Gd': 4 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'BsmtFinType1'
    var_map =  { 'None': -6, 'Unf':-4, 'Rec':-2, 'LwQ':0, 'BLQ': 1, 'ALQ':2, 'GLQ':3 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'BsmtFinType2'
    var_map =  { 'None': -6, 'Unf':-4, 'Rec':-2, 'LwQ':0, 'BLQ': 1, 'ALQ':2, 'GLQ':3 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'Heating'
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)


    var = 'HeatingQC'
    var_map =  { 'Po':1, 'Fa':2, 'TA':3, 'Gd': 4, 'Ex':5 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'CentralAir'
    clean_ALL[var] = clean_ALL[var].apply(lambda x: 0 if x == 'N' else 1)


    var = 'Electrical'
    mask = clean_ALL[var] == 'Mix'
    clean_ALL.loc[mask, var] = "FuseP"
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)

    #
    var = 'KitchenQual'
    var_map =  { 'Po':1, 'Fa':2, 'TA':3, 'Gd': 4, 'Ex':5 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'Functional'
    var_map =  { 'Sev':0, 'Maj2':1, 'Maj1':2, 'Mod':3, 'Min2':4, 'Min1':5, 'Typ':6 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'FireplaceQu'
    var_map =  { 'None':-2, 'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'GarageType'
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)


    var = 'GarageFinish'
    var_map =  { 'None':-1, 'Unf':1, 'RFn':2, 'Fin':3 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'GarageQual'
    var_map =  { 'None':-2, 'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'GarageCond'
    var_map =  { 'None':-2, 'Po':0, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'PavedDrive'
    var_map =  { 'N':0, 'P':1, 'Y':2 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'PoolQC'
    var_map =  { 'None':-1, 'Fa':1, 'TA':2, 'Gd':3, 'Ex':4 }
    clean_ALL[var] = clean_ALL[var].map(var_map)


    var = 'MiscFeature'
    mask = clean_ALL[var] == 'RL'
    clean_ALL.loc[mask, var] = "None"
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)


    var = 'SaleType'
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)


    var = 'MoSold'
    month_map = { 1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec' }
    clean_ALL[var] = clean_ALL[var].map(month_map)
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)


    var = 'SaleCondition'
    clean_ALL = pd.get_dummies(clean_ALL, columns=[var], prefix=var, drop_first=False)


    return clean_ALL
