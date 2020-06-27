########################### import section
#################################################################################
import pandas as pd
import numpy as np
from math import ceil
import time
import os, sys, gc, time, warnings, pickle, psutil, random
from multiprocessing import Pool        # Multiprocess Runs
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from module.prepare_data import *
from catboost import CatBoostRegressor

cat_params = {
    'n_estimators':1400,
    'loss_function':'Tweedie',
    # 'tweedie_variance_power': 1.1,
    'eval_metric':'RMSE',
    'subsample': 0.5,
    'sampling_frequency':1,
    'learning_rate':0.03,
    'max_leaves': 2 ** 11 - 1,
    'min_data_in_leaf': 2 ** 12 - 1,
    'feature_fraction': 0.5,
    'max_bin': 100,
    'verbose': 1,
    'random_seed': SEED,
}


def train_evaluate_model(feature_columns, target, base_path, stores_ids=STORES_IDS, permutation=False):

    his = []
    for store_id in stores_ids:
        print('Train', store_id)

        grid_df = get_data_by_store(store_id)
        train_mask = grid_df['d'] <= END_TRAIN
        # valid_mask = (grid_df['d'] > END_TRAIN-28 -100) & (grid_df['d'] <= END_TRAIN)
        preds_mask = grid_df['d'] > (END_TRAIN - 100)

        ## Initiating our GroupKFold
        folds = GroupKFold(n_splits=3)
        # grid_df['groups'] = grid_df['tm_y'].astype(str) + '_' + grid_df['tm_m'].astype(str)
        split_groups = grid_df[train_mask]['groups']

        # Saving part of the dataset for later predictions
        # Removing features that we need to calculate recursively
        keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
        grid_df[preds_mask].reset_index(drop=True)[keep_cols].to_pickle(f'{base_path}/test_{store_id}_ver{VER}.pkl')
        # grid_df[valid_mask].reset_index(drop=True)[keep_cols].to_pickle(f'{base_path}/valid_{store_id}_ver{VER}.pkl')

        feature_columns_i = feature_columns[store_id]
        # Main Data
        X, y = grid_df[train_mask][feature_columns_i], grid_df[train_mask][target]
        del grid_df


        # Launch seeder again to make lgb training 100% deterministic
        # with each "code line" np.random "evolves"
        # so we need (may want) to "reset" it

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)):

            print('Fold:', fold_)
            trn_X, trn_y = X.iloc[trn_idx, :], y[trn_idx]
            val_X, val_y = X.iloc[val_idx, :], y[val_idx]
            # train_data = lgb.Dataset(trn_X, label=trn_y)
            # valid_data = lgb.Dataset(val_X, label=val_y)
            estimator = CatBoostRegressor(**cat_params)
            estimator = estimator.fit(trn_X, trn_y, eval_set=(trn_y, val_y), silent=True)

            if permutation:
                importance_df = permutation_importance(estimator, pd.concat([val_X,val_y], axis=1), feature_columns_i, target, metric=root_mean_sqared_error,verbose=0)
            else:
                importance_df = None

            prediction_val = estimator.predict(val_X)
            rmse_val = rmse(val_y, prediction_val)
            prediction_trn = estimator.predict(trn_X)
            rmse_trn = rmse(trn_y, prediction_trn)

            # Save model - it's not real '.bin' but a pickle file
            # estimator = lgb.Booster(model_file='model.txt')
            # can only predict with the best iteration (or the saving iteration)
            # pickle.dump gives us more flexibility
            # like estimator.predict(TEST, num_iteration=100)
            # num_iteration - number of iteration want to predict with,
            # NULL or <= 0 means use best iteration
            model_name = f'{base_path}/lgb_model_{store_id}_fold{fold_}_ver{VER}.bin'
            pickle.dump(estimator, open(model_name, 'wb'))

            # Remove temporary files and objects
            # to free some hdd space and ram memory
            del estimator, trn_X, val_X, trn_y, val_y
            gc.collect()

            his.append({'rmse_val': rmse_val, 'rmse_trn':rmse_trn, 'rmse_diff':rmse_val-rmse_trn, 'fold_': fold_, 'store_id': store_id, 'prediction_val':prediction_val, 'permutation_importance':importance_df})

    return pd.DataFrame(his)

def predict_test(feature_columns, target, base_path, stores_ids=STORES_IDS, key='test', end_train=END_TRAIN):

    pridiction_list = []
    if key == 'valid':
        end_train -= 28

    for fold_ in range(CV_FOLDS):
        all_preds = pd.DataFrame()
        base_test = get_base_test(base_path, stores_ids, key=key)
        main_time = time.time()

        for PREDICT_DAY in range(1, 29):
            print(f'FOLD{fold_} Predict | Day:{PREDICT_DAY}')
            start_time = time.time()

            # Make temporary grid to calculate rolling lags
            grid_df = base_test.copy()
            grid_df = extract_sliding_shift_features(grid_df, target)

            for store_id in stores_ids:
                feature_columns_i = feature_columns[store_id]
                model_name = f'{base_path}/lgb_model_{store_id}_fold{fold_}_ver{VER}.bin'
                estimator = pickle.load(open(model_name, 'rb'))

                day_mask = base_test['d'] == (end_train + PREDICT_DAY)
                store_mask = base_test['store_id'] == store_id
                mask = (day_mask) & (store_mask)
                base_test.loc[mask, target] = estimator.predict(grid_df[mask][feature_columns_i])

            # Make good column naming and add
            # to all_preds DataFrame
            temp_df = base_test[day_mask][['id', target]]
            temp_df.columns = ['id', 'F' + str(PREDICT_DAY)]

            if 'id' in list(all_preds):
                all_preds = all_preds.merge(temp_df, on=['id'], how='left')
            else:
                all_preds = temp_df.copy()
            all_preds = all_preds.reset_index(drop=True)
            print('#' * 10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                  ' %0.2f min total |' % ((time.time() - main_time) / 60),
                  ' %0.2f day sales |' % (temp_df['F' + str(PREDICT_DAY)].sum()))
        pridiction_list.append(all_preds)

    final_all_preds = pd.DataFrame()
    final_all_preds['id'] = pridiction_list[0]['id']
    for item in pridiction_list[0]:
        if item != 'id':
            final_all_preds[item] = (pridiction_list[0][item] * (1 / 3)) + (pridiction_list[1][item] * (1 / 3)) + (pridiction_list[2][item] * (1 / 3))

    if key == 'valid':
        melt_preds_df = pd.melt(final_all_preds, id_vars=['id'], var_name='d', value_name=target)
        melt_preds_df.replace({'d': dict(zip([f'F{i}' for i in range(1, 29)], [END_TRAIN - 28 + i for i in range(1, 29)]))}, inplace=True)
        melt_preds_df = pd.merge(melt_preds_df, get_base_test(BASE_PATH, stores_ids, key=key)[['id', 'd', 'sales']], on=['id', 'd'], how='left')
        rmse_val = rmse(melt_preds_df['sales_y'].values, melt_preds_df['sales_x'].values)
        print('rmse_val', rmse_val)

    return final_all_preds