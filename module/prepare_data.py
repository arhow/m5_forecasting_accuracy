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
from pandarallel import pandarallel
pandarallel.initialize()

########################### globle var section
#################################################################################
VER =620
SEED = 42                        # We want all things
ORI_CSV_PATH = '../input/m5-forecasting-accuracy2'
STORES_IDS = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
TARGET = 'sales'
START_TRAIN = 0                  # We can skip some rows (Nans/faster training)
END_TRAIN   = 1913+28               # End day of our train set
# P_HORIZON   = 28                 # Prediction horizon
# USE_AUX     = False               # Use or not pretrained models
BASE_PATH  = f'../cache/ver{VER}'
TEMP_FEATURE_PKL =  f'{BASE_PATH}/grid_features.pkl'

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

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def root_mean_sqared_error(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

def permutation_importance(model, validation_df, features_columns, target, metric=root_mean_sqared_error, verbose=0):

    list_ = []
    # Make normal prediction with our model and save score
    validation_df['preds'] = model.predict(validation_df[features_columns])
    base_score = metric(validation_df[target], validation_df['preds'])
    if verbose > 0:
        print('Standart RMSE', base_score)

    # Now we are looping over all our numerical features
    for col in features_columns:

        # We will make validation set copy to restore
        # features states on each run
        temp_df = validation_df.copy()

        # Error here appears if we have "categorical" features and can't
        # do np.random.permutation without disrupt categories
        # so we need to check if feature is numerical
        if temp_df[col].dtypes.name != 'category':
            temp_df[col] = np.random.permutation(temp_df[col].values)
            temp_df['preds'] = model.predict(temp_df[features_columns])
            cur_score = metric(temp_df[target], temp_df['preds'])

            list_.append({'feature': col, 'permutation_importance': np.round(cur_score - base_score, 4)})
            # If our current rmse score is less than base score
            # it means that feature most probably is a bad one
            # and our model is learning on noise
            if verbose > 0:
                print(col, np.round(cur_score - base_score, 4))

    return pd.DataFrame(list_).sort_values(by=['permutation_importance'], ascending=False)

########################### feature extract
#################################################################################
def extract_features(train_df, prices_df, calendar_df, target, nan_mask_d):
    stop_watch = [time.time()]
    grid_df = melt_train_df(train_df, prices_df, calendar_df, target)
    stop_watch.append(time.time())
    print(f'melt_train_df {stop_watch[-1]-stop_watch[-2]}')
    grid_df = extract_price_features(prices_df, calendar_df, grid_df)
    stop_watch.append(time.time())
    print(f'extract_price_features {stop_watch[-1]-stop_watch[-2]}')
    grid_df = extract_calendar_features(calendar_df, grid_df)
    stop_watch.append(time.time())
    print(f'extract_calendar_features {stop_watch[-1]-stop_watch[-2]}')
    grid_df = extract_rolling_features(grid_df, target)
    stop_watch.append(time.time())
    print(f'extract_rolling_features {stop_watch[-1]-stop_watch[-2]}')
    grid_df = extract_encode_features(grid_df, target, nan_mask_d)
    stop_watch.append(time.time())
    print(f'extract_encode_features {stop_watch[-1]-stop_watch[-2]}')
    grid_df = extract_sliding_shift_features(grid_df, target)
    stop_watch.append(time.time())
    print(f'extract_sliding_shift_features {stop_watch[-1]-stop_watch[-2]}')
    return grid_df

def get_data_by_store(store_id, grid_df_path = TEMP_FEATURE_PKL):
    grid_df = pd.read_pickle(grid_df_path)
    for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
        n_nan = grid_df[col].isna().sum()
        if n_nan > 0:
            grid_df[col] = grid_df[col].astype(str).fillna('NONE').astype('category')
    return grid_df[grid_df['store_id']==store_id].reset_index(drop=False)


def melt_train_df(train_df, prices_df, calendar_df, target):
    index_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    grid_df = pd.melt(train_df,
                      id_vars=index_columns,
                      var_name='d',
                      value_name=target)

    # To be able to make predictions
    # we need to add "test set" to our grid
    add_grid = pd.DataFrame()
    for i in range(1, 29):
        temp_df = train_df[index_columns]
        temp_df = temp_df.drop_duplicates()
        temp_df['d'] = 'd_' + str(END_TRAIN + i)
        temp_df[TARGET] = np.nan
        add_grid = pd.concat([add_grid, temp_df])

    grid_df = grid_df[~grid_df['d'].isin(add_grid['d'].unique().tolist())]

    grid_df = pd.concat([grid_df, add_grid])
    grid_df = grid_df.reset_index(drop=True)
    del add_grid

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
    grid_df['release'] = grid_df['release'] - grid_df['release'].min()
    grid_df['release'] = grid_df['release'].astype(np.int16)

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
    # global  base_test
    # base_test = grid_df[['id','d',TARGET]]
    old_tmp_cols = [col for col in list(grid_df) if '_tmp_' in col]
    grid_df = grid_df.drop(columns=old_tmp_cols)
    ROLS_SPLIT = []
    for i in [1, 7, 14]:
        for j in [7, 14, 30, 60]:
            ROLS_SPLIT.append([i, j])
#     N_CORES = psutil.cpu_count()
#     from joblib import Parallel, delayed
#     lst = Parallel(n_jobs=N_CORES)(delayed(_make_lag_roll)(grid_df[['id','d',target]], target, item[0], item[1]) for item in ROLS_SPLIT)
#     grid_df_add = pd.concat(lst, axis=1)
#     grid_df = pd.concat([grid_df,grid_df_add], axis=1)
    for item in ROLS_SPLIT:
        grid_df_add = _make_lag_roll(grid_df[['id','d',target]], target, item[0], item[1])
        grid_df = pd.concat([grid_df,grid_df_add], axis=1)
    return grid_df


def extract_encode_features(grid_df, target, nan_mask_d):
    base_ = grid_df[['store_id', 'cat_id', 'dept_id', 'item_id', 'tm_dw', 'tm_w_end', target]].copy()
    base_.loc[grid_df['d'] > nan_mask_d, target] = np.nan
    icols = [
        ['cat_id'],
        ['dept_id'],
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

def _make_lag_roll(base_test, target, shift_day, roll_wind):
    # target, shift_day, roll_wind = LAG_DAY[0],LAG_DAY[1],LAG_DAY[2]
    lag_df = base_test[['id','d',target]]
    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
    # lag_df[col_name] = lag_df.groupby(['id'])[target].rolling(roll_wind).parallel_apply(np.mean).reset_index(0, drop=True)
    # lag_df[col_name] = lag_df.groupby(['id'])[col_name].transform(lambda x: x.shift(shift_day))
    lag_df[col_name] = lag_df.groupby(['id'])[target].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean()).astype(np.float16)
    return lag_df[[col_name]]

def get_base_test(base_path, stores_ids=STORES_IDS, key='test'):
    base_test = pd.DataFrame()
    for store_id in stores_ids:
        temp_df = pd.read_pickle(f'{base_path}/{key}_{store_id}_ver{VER}.pkl')
        temp_df['store_id'] = store_id
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
    for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
        n_nan = base_test[col].isna().sum()
        if n_nan > 0:
            base_test[col] = base_test[col].astype(str).fillna('NONE').astype('category')

    return base_test






