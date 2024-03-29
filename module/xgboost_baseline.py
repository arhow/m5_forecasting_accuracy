########################### import section
#################################################################################
import pandas as pd
import numpy as np
from math import ceil
import time
import os, sys, gc, time, warnings, pickle, psutil, random
from multiprocessing import Pool        # Multiprocess Runs
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from module.prepare_data import *

xgb_params = {
                    'booster': 'gbtree',
                    'objective': 'reg:tweedie',
                    'tweedie_variance_power': 1.1,
                    'eval_metric': 'rmse',
                    'subsample': 0.5,
                    'eta': 0.03,
                    'max_depth': 9,
                    'max_bin': 100,
                    'verbosity': 1,
                    'seed':SEED,
# 'tree_method':'gpu_hist', 'gpu_id':0, 'task_type':"GPU",
                }



def train_evaluate_model(feature_columns, target, base_path, stores_ids=STORES_IDS, permutation=False):

    his = []
    for store_id in stores_ids:
        print('Train', store_id)

        grid_df = get_data_by_store(store_id)
        grid_df['item_id'] = grid_df['item_id'].apply(lambda x: int(x.split('_')[-1])).astype('category')
        grid_df['dept_id'] = grid_df['dept_id'].apply(lambda x: int(x.split('_')[-1])).astype('category')
        grid_df['cat_id'] = grid_df['cat_id'].replace({'HOBBIES': 0, 'HOUSEHOLD': 1, 'FOODS': 2}).astype('category')
        for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
            grid_df[col] = grid_df[col].replace(
                dict(zip(grid_df[col].unique(), np.arange(grid_df[col].unique().shape[0])))).astype('category')

        train_mask = grid_df['d'] <= END_TRAIN
        preds_mask = grid_df['d'] > (END_TRAIN - 100)

        ## Initiating our GroupKFold
        folds = GroupKFold(n_splits=3)
        grid_df['groups'] = grid_df['tm_y'].astype(str)
        split_groups = grid_df[train_mask]['groups']

        feature_columns_i = feature_columns[store_id]
        # Main Data
        X, y = grid_df[train_mask][feature_columns_i], grid_df[train_mask][target]

        # Saving part of the dataset for later predictions
        # Removing features that we need to calculate recursively
        keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
        if not os.path.exists(f'{base_path}/test_{store_id}_ver{VER}.pkl'):
            grid_df[preds_mask].reset_index(drop=True)[keep_cols].to_pickle(f'{base_path}/test_{store_id}_ver{VER}.pkl')
        del grid_df

        # Launch seeder again to make lgb training 100% deterministic
        # with each "code line" np.random "evolves"
        # so we need (may want) to "reset" it

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)):

            print('Fold:', fold_)
            trn_X, trn_y = X.iloc[trn_idx, :], y[trn_idx]
            val_X, val_y = X.iloc[val_idx, :], y[val_idx]
            train_data = xgb.DMatrix(trn_X.values, label=trn_y.values)
            valid_data = xgb.DMatrix(val_X.values, label=val_y.values)
            estimator = xgb.train(xgb_params, train_data, num_boost_round=500, evals=[(train_data,'train'), (valid_data,'eval')], verbose_eval=100)

            if permutation:
                importance_df = permutation_importance(estimator, pd.concat([val_X, val_y], axis=1), feature_columns_i,
                                                       target, metric=root_mean_sqared_error, verbose=0)
            else:
                importance_df = None

            prediction_val = estimator.predict(valid_data)
            rmse_val = rmse(val_y, prediction_val)
            prediction_trn = estimator.predict(train_data)
            rmse_trn = rmse(trn_y, prediction_trn)

            # Save model - it's not real '.bin' but a pickle file
            # estimator = lgb.Booster(model_file='model.txt')
            # can only predict with the best iteration (or the saving iteration)
            # pickle.dump gives us more flexibility
            # like estimator.predict(TEST, num_iteration=100)
            # num_iteration - number of iteration want to predict with,
            # NULL or <= 0 means use best iteration
            model_name = f'{base_path}/xgboost_model_{store_id}_fold{fold_}_ver{VER}.bin'
            pickle.dump(estimator, open(model_name, 'wb'))

            # Remove temporary files and objects
            # to free some hdd space and ram memory
            del train_data, valid_data, estimator, trn_X, val_X, trn_y, val_y
            gc.collect()

            his.append({'rmse_val': rmse_val, 'rmse_trn': rmse_trn, 'rmse_diff': rmse_val - rmse_trn, 'fold_': fold_,
                        'store_id': store_id, 'prediction_val': prediction_val,
                        'permutation_importance': importance_df})

    return pd.DataFrame(his)



def predict_test(feature_columns, target, base_path, stores_ids=STORES_IDS):

    pridiction_list = []
    for fold_ in range(CV_FOLDS):
        all_preds = pd.DataFrame()
        base_test = get_base_test(base_path, stores_ids)
        base_test['item_id'] = base_test['item_id'].apply(lambda x: int(x.split('_')[-1])).astype('category')
        base_test['dept_id'] = base_test['dept_id'].apply(lambda x: int(x.split('_')[-1])).astype('category')
        base_test['cat_id'] = base_test['cat_id'].replace({'HOBBIES': 0, 'HOUSEHOLD': 1, 'FOODS': 2}).astype('category')
        for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
            base_test[col] = base_test[col].replace(
                dict(zip(base_test[col].unique(), np.arange(base_test[col].unique().shape[0])))).astype('category')
        main_time = time.time()

        for PREDICT_DAY in range(1, 29):
            print(f'FOLD{fold_} Predict | Day:{PREDICT_DAY}')
            start_time = time.time()

            # Make temporary grid to calculate rolling lags
            grid_df = base_test.copy()
            grid_df = extract_sliding_shift_features(grid_df, target)

            for store_id in stores_ids:
                feature_columns_i = feature_columns[store_id]
                model_name = f'{base_path}/xgboost_model_{store_id}_fold{fold_}_ver{VER}.bin'
                estimator = pickle.load(open(model_name, 'rb'))

                day_mask = base_test['d'] == (END_TRAIN + PREDICT_DAY)
                store_mask = base_test['store_id'] == store_id
                mask = (day_mask) & (store_mask)
                dMatrix = xgb.DMatrix(grid_df[mask][feature_columns_i].values)
                base_test.loc[mask, target] = estimator.predict(dMatrix)

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

    return final_all_preds


# def permutation(features_columns, target, base_path, stores_ids=STORES_IDS):
#     his = []
#     for store_id in stores_ids:
#         print('permutation', store_id)
#
#         grid_df = get_data_by_store(store_id)
#
#         train_mask = grid_df['d'] <= END_TRAIN
#
#         ## Initiating our GroupKFold
#         folds = GroupKFold(n_splits=3)
#         grid_df['groups'] = grid_df['tm_y'].astype(str) + '_' + grid_df['tm_m'].astype(str)
#         split_groups = grid_df[train_mask]['groups']
#         X, y = grid_df[train_mask][features_columns].reset_index(drop=True), grid_df[train_mask][target].reset_index(
#             drop=True)
#         del grid_df
#
#         for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)):
#             val_X, val_y = X.iloc[val_idx, :], y[val_idx]
#             print('Fold:', fold_)
#             model_name = f'{base_path}/lgb_model_{store_id}_fold{fold_}_ver{VER}.bin'
#             estimator = pickle.load(open(model_name, 'rb'))
#             permutation_importance_df = permutation_importance(estimator, pd.concat([val_X, val_y], axis=1),
#                                                                features_columns, target, metric=root_mean_sqared_error,
#                                                                verbose=0)
#             del estimator, val_X, val_y
#             gc.collect()
#
#             his.append({'permutation_importance_df': permutation_importance_df, 'fold_': fold_, 'store_id': store_id})
#
#     return pd.DataFrame(his)

