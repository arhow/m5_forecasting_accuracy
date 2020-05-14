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

########################### globle var section
#################################################################################
VER =10
SEED = 42                        # We want all things
ORI_CSV_PATH = '../input/m5-forecasting-accuracy'
STORES_IDS = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
TARGET = 'sales'
START_TRAIN = 0                  # We can skip some rows (Nans/faster training)
END_TRAIN   = 1913               # End day of our train set
P_HORIZON   = 28                 # Prediction horizon
USE_AUX     = False               # Use or not pretrained models
BASE_PATH  = f'../cache/ver{VER}'
TEMP_FEATURE_PKL =  f'{BASE_PATH}/grid_features.pkl'
lgb_params = {
                    'boosting_type': 'gbdt',
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'subsample': 0.5,
                    'subsample_freq': 1,
                    'learning_rate': 0.03,
                    'num_leaves': 2**11-1,
                    'min_data_in_leaf': 2**12-1,
                    'feature_fraction': 0.5,
                    'max_bin': 100,
                    'n_estimators': 1400,
                    'boost_from_average': False,
                    'verbose': 1,
                    'seed':SEED,
                }
M5_FEATURES = ['item_id',
 'dept_id',
 'cat_id',
 'release',
 'sell_price',
 'price_max',
 'price_min',
 'price_std',
 'price_mean',
 'price_norm',
 'price_nunique',
 'item_nunique',
 'price_momentum',
 'price_momentum_m',
 'price_momentum_y',
 'event_name_1',
 'event_type_1',
 'event_name_2',
 'event_type_2',
 'snap_CA',
 'snap_TX',
 'snap_WI',
 'tm_d',
 'tm_w',
 'tm_m',
 'tm_y',
 'tm_wm',
 'tm_dw',
 'tm_w_end',
 'enc_cat_id_mean',
 'enc_cat_id_std',
 'enc_dept_id_mean',
 'enc_dept_id_std',
 'enc_item_id_mean',
 'enc_item_id_std',
 'sales_lag_28',
 'sales_lag_29',
 'sales_lag_30',
 'sales_lag_31',
 'sales_lag_32',
 'sales_lag_33',
 'sales_lag_34',
 'sales_lag_35',
 'sales_lag_36',
 'sales_lag_37',
 'sales_lag_38',
 'sales_lag_39',
 'sales_lag_40',
 'sales_lag_41',
 'sales_lag_42',
 'rolling_mean_7',
 'rolling_std_7',
 'rolling_mean_14',
 'rolling_std_14',
 'rolling_mean_30',
 'rolling_std_30',
 'rolling_mean_60',
 'rolling_std_60',
 'rolling_mean_180',
 'rolling_std_180',
 'rolling_mean_tmp_1_7',
 'rolling_mean_tmp_1_14',
 'rolling_mean_tmp_1_30',
 'rolling_mean_tmp_1_60',
 'rolling_mean_tmp_7_7',
 'rolling_mean_tmp_7_14',
 'rolling_mean_tmp_7_30',
 'rolling_mean_tmp_7_60',
 'rolling_mean_tmp_14_7',
 'rolling_mean_tmp_14_14',
 'rolling_mean_tmp_14_30',
 'rolling_mean_tmp_14_60']

CV_FOLDS = 3

########################### Utility
#################################################################################
def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1

########################### feature extract
#################################################################################
def extract_features(train_df, prices_df, calendar_df, target, nan_mask_d=1913-28):

    grid_df = melt_train_df(train_df, prices_df, calendar_df, target)
    grid_df = extract_price_features(prices_df, calendar_df, grid_df)
    grid_df = extract_calendar_features(calendar_df, grid_df)
    grid_df = extract_rolling_features(grid_df, target)
    grid_df = extract_encode_features(grid_df, target, nan_mask_d)
    grid_df = extract_sliding_shift_features(grid_df, target)
    return grid_df


def melt_train_df(train_df, prices_df, calendar_df, target):
    index_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    grid_df = pd.melt(train_df,
                      id_vars=index_columns,
                      var_name='d',
                      value_name=target)

    # Prices are set by week
    # so it we will have not very accurate release week
    release_df = prices_df.groupby(['store_id', 'item_id'])['wm_yr_wk'].agg(['min']).reset_index()
    release_df.columns = ['store_id', 'item_id', 'release']

    # Now we can merge release_df
    grid_df = merge_by_concat(grid_df, release_df, ['store_id', 'item_id'])
    # del release_df

    # We want to remove some "zeros" rows
    # from grid_df
    # to do it we need wm_yr_wk column
    # let's merge partly calendar_df to have it
    grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk', 'd']], ['d'])

    # Now we can cutoff some rows
    # and safe memory
    grid_df = grid_df[grid_df['wm_yr_wk'] >= grid_df['release']]
    grid_df = grid_df.reset_index(drop=True)
    return grid_df

def extract_price_features(prices_df, calendar_df, grid_df):

    # We can do some basic aggregations
    prices_df['price_max'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('max')
    prices_df['price_min'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('min')
    prices_df['price_std'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('std')
    prices_df['price_mean'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('mean')

    # and do price normalization (min/max scaling)
    prices_df['price_norm'] = prices_df['sell_price']/prices_df['price_max']

    # Some items are can be inflation dependent
    # and some items are very "stable"
    prices_df['price_nunique'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')
    prices_df['item_nunique'] = prices_df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')

    # I would like some "rolling" aggregations
    # but would like months and years as "window"
    calendar_prices = calendar_df[['wm_yr_wk','month','year']]
    calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
    prices_df = prices_df.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')

    # Now we can add price "momentum" (some sort of)
    # Shifted by week
    # by month mean
    # by year mean
    prices_df['price_momentum'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))
    prices_df['price_momentum_m'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
    prices_df['price_momentum_y'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')

    del prices_df['month'], prices_df['year']
    grid_df = grid_df.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')

    return grid_df

def extract_calendar_features(calendar_df, grid_df):

    # Merge calendar partly
    icols = ['date',
             'd',
             'event_name_1',
             'event_type_1',
             'event_name_2',
             'event_type_2',
             'snap_CA',
             'snap_TX',
             'snap_WI']

    grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')
    grid_df['d'] = grid_df['d'].apply(lambda x: int(x[2:]))

    # Minify data
    # 'snap_' columns we can convert to bool or int8
    icols = ['event_name_1',
             'event_type_1',
             'event_name_2',
             'event_type_2',
             'snap_CA',
             'snap_TX',
             'snap_WI']
    for col in icols:
        grid_df[col] = grid_df[col].astype('category')

    # Convert to DateTime
    grid_df['date'] = pd.to_datetime(grid_df['date'])

    # Make some features from date
    grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)
    grid_df['tm_w'] = grid_df['date'].dt.week.astype(np.int8)
    grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)
    grid_df['tm_y'] = grid_df['date'].dt.year
    grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)
    grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int8)
    grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8)
    grid_df['tm_w_end'] = (grid_df['tm_dw']>=5).astype(np.int8)
    del grid_df['date']
    return grid_df

def extract_rolling_features(grid_df, target, shift_day=28, verbose=0):

    grid_df_add = grid_df[['id', 'd', 'sales']]
    # Lags
    # with 28 day shift
    start_time = time.time()
    if verbose > 0:
        print('Create lags')
    LAG_DAYS = [col for col in range(shift_day, shift_day + 15)]
    grid_df_add = grid_df_add.assign(**{
        '{}_lag_{}'.format(col, l): grid_df_add.groupby(['id'])[col].transform(lambda x: x.shift(l))
        for l in LAG_DAYS
        for col in [target]
    })

    # Minify lag columns
    for col in list(grid_df_add):
        if 'lag' in col:
            grid_df_add[col] = grid_df_add[col].astype(np.float16)
    if verbose > 0:
        print('%0.2f min: Lags' % ((time.time() - start_time) / 60))

    # Rollings
    # with 28 day shift
    start_time = time.time()
    if verbose > 0:
        print('Create rolling aggs')

    for i in [7, 14, 30, 60, 180]:
        if verbose > 0:
            print('Rolling period:', i)
        grid_df_add['rolling_mean_' + str(i)] = grid_df_add.groupby(['id'])[target].transform(
            lambda x: x.shift(shift_day).rolling(i).mean()).astype(np.float16)
        grid_df_add['rolling_std_' + str(i)] = grid_df_add.groupby(['id'])[target].transform(
            lambda x: x.shift(shift_day).rolling(i).std()).astype(np.float16)

    grid_df = merge_by_concat(grid_df, grid_df_add.drop(columns=['sales']), ['id', 'd'])
    return grid_df

def extract_sliding_shift_features(grid_df, target):
    # base_test = grid_df[['id','d',TARGET]]
    old_tmp_cols = [col for col in list(grid_df) if '_tmp_' in col]
    grid_df = grid_df.drop(columns=old_tmp_cols)
    print(grid_df.shape)
    ROLS_SPLIT = []
    for i in [1, 7, 14]:
        for j in [7, 14, 30, 60]:
            ROLS_SPLIT.append([grid_df, target, i, j])
    grid_df = pd.concat([grid_df, _df_parallelize_run(_make_lag_roll, ROLS_SPLIT)], axis=1)
    return grid_df


def extract_encode_features(grid_df, target, nan_mask_d):
    base_ = grid_df[['store_id', 'cat_id', 'dept_id', 'item_id', 'tm_dw', TARGET]].copy()
    base_[TARGET][grid_df['d'] > nan_mask_d] = np.nan
    icols = [
        ['cat_id'],
        ['dept_id'],
        ['store_id'],
        ['item_id'],
    ]
    for col in icols:
        col_name = '_' + '_'.join(col) + '_'
        grid_df['enc' + col_name + 'mean'] = base_.groupby(col)[target].transform('mean').astype(np.float16)
        grid_df['enc' + col_name + 'std'] = base_.groupby(col)[target].transform('std').astype(np.float16)
    return grid_df


## Multiprocess Runs
def _df_parallelize_run(func, t_split):
    N_CORES = psutil.cpu_count()  # Available CPU cores
    num_cores = np.min([N_CORES,len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df

def _make_lag_roll(LAG_DAY):
    base_test, target, shift_day, roll_wind = LAG_DAY[0],LAG_DAY[1],LAG_DAY[2],LAG_DAY[3]
    lag_df = base_test[['id','d',target]]
    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])[target].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return lag_df[[col_name]]


def train_evaluate_model(grid_df, feature_columns, target, base_path):

    his = []
    for store_id in STORES_IDS:
        print('Train', store_id)
        train_mask = grid_df['d'] <= END_TRAIN
        preds_mask = grid_df['d'] > (END_TRAIN - 100)

        ## Initiating our GroupKFold
        folds = GroupKFold(n_splits=3)
        grid_df['groups'] = grid_df['tm_w'].astype(str) + '_' + grid_df['tm_y'].astype(str)
        split_groups = grid_df[train_mask]['groups']

        # Main Data
        X, y = grid_df[train_mask][feature_columns], grid_df[train_mask][target]

        # Saving part of the dataset for later predictions
        # Removing features that we need to calculate recursively
        grid_df = grid_df[preds_mask].reset_index(drop=True)
        keep_cols = [col for col in list(grid_df) if '_tmp_' not in col]
        grid_df = grid_df[keep_cols]
        grid_df.to_pickle(f'{base_path}/test_{store_id}_ver{VER}.pkl')


        # Launch seeder again to make lgb training 100% deterministic
        # with each "code line" np.random "evolves"
        # so we need (may want) to "reset" it

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)):

            print('Fold:', fold_)
            trn_X, trn_y = X.iloc[trn_idx, :], y[trn_idx]
            val_X, val_y = X.iloc[val_idx, :], y[val_idx]
            train_data = lgb.Dataset(trn_X, label=trn_y)
            valid_data = lgb.Dataset(val_X, label=val_y)
            estimator = lgb.train(lgb_params, train_data, valid_sets=[train_data, valid_data], verbose_eval=100)
            prediction_val = estimator.predict(val_X)
            rmse_val = rmse(val_y, prediction_val)

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
            del train_data, valid_data, estimator
            gc.collect()

            his.append({'rmse_val': rmse_val, 'fold_': fold_, 'store_id': store_id})

    return pd.DataFrame(his)

def get_base_test(base_path):
    base_test = pd.DataFrame()
    for store_id in STORES_IDS:
        temp_df = pd.read_pickle(f'{base_path}/test_{store_id}_ver{VER}.pkl')
        temp_df['store_id'] = store_id
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)

    return base_test

    return

def predict_test(grid_df, feature_columns, target, base_path):

    pridiction_list = []
    for fold_ in CV_FOLDS:
        all_preds = pd.DataFrame()
        for PREDICT_DAY in range(1, 29):
            print('Predict | Day:', PREDICT_DAY)
            start_time = time.time()

            # Make temporary grid to calculate rolling lags
            grid_df = grid_df.copy()
            grid_df = extract_sliding_shift_features(grid_df, target)

            for store_id in STORES_IDS:

                model_name = f'{base_path}/lgb_model_{store_id}_fold{fold_}_ver{VER}.bin'
                estimator = pickle.load(open(model_name, 'rb'))

                day_mask = grid_df['d'] == (END_TRAIN + PREDICT_DAY)
                store_mask = grid_df['store_id'] == store_id
                mask = (day_mask) & (store_mask)
                grid_df[target][mask] = estimator.predict(grid_df[mask][feature_columns])

            # Make good column naming and add
            # to all_preds DataFrame
            temp_df = grid_df[day_mask][['id', target]]
            temp_df.columns = ['id', 'F' + str(PREDICT_DAY)]
            if 'id' in list(all_preds):
                all_preds = all_preds.merge(temp_df, on=['id'], how='left')
            else:
                all_preds = temp_df.copy()
            pridiction_list.append(all_preds.reset_index(drop=True))

    final_all_preds = pd.DataFrame()
    final_all_preds['id'] = pridiction_list[0]['id']
    for item in pridiction_list[0]:
        if item != 'id':
            final_all_preds[item] = (pridiction_list[0][item] * (1 / 3)) + (pridiction_list[1][item] * (1 / 3)) + (pridiction_list[2][item] * (1 / 3))

    return final_all_preds


########################### RUN
#################################################################################
# train_df = pd.read_csv(f'{ORI_CSV_PATH}/sales_train_validation.csv')
# prices_df = pd.read_csv(f'{ORI_CSV_PATH}/sell_prices.csv')
# calendar_df = pd.read_csv(f'{ORI_CSV_PATH}/calendar.csv')
#
# try:
#     os.makedirs(BASE_PATH)
# except OSError:
#     print ("Creation of the directory %s failed" % BASE_PATH)
# else:
#     print ("Successfully created the directory %s" % BASE_PATH)
#
# if not os.path.exists(TEMP_FEATURE_PKL):
#     grid_df = extract_features(train_df, prices_df, calendar_df, target=TARGET, nan_mask_d=1913-28)
#     grid_df.to_pickle(TEMP_FEATURE_PKL)
# else:
#     grid_df = pd.read_pickle(TEMP_FEATURE_PKL)
#
# history_df = train_evaluate_model(grid_df, M5_FEATURES, TARGET, BASE_PATH)
# history_df.to_pickle(f'{BASE_PATH}/history.pkl')
#
# base_test = get_base_test(BASE_PATH)
#
# final_all_preds = predict_test(base_test, M5_FEATURES, TARGET, BASE_PATH)
# history_df.to_csv(f'{BASE_PATH}/submission.csv', index=False)

