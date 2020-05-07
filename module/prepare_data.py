import numpy as np
import pandas as pd
import time
import os, sys, gc, time, warnings, pickle, psutil, random
from multiprocessing import Pool        # Multiprocess Runs

## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
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


def extract_features(base_path, Target):
    grid_df = pd.concat([pd.read_pickle(f'{base_path}/ori_grid_part_1.pkl'),
                         pd.read_pickle(f'{base_path}/ori_grid_part_2.pkl').iloc[:,2:],
                         pd.read_pickle(f'{base_path}/ori_grid_part_3.pkl').iloc[:,2:]],
                         axis=1)
    grid_df = gen_encode_features(grid_df, Target)
    return grid_df

def load_base_features(read_base_path, save_base_path, TARGET):
    if os.path.exists(f'{save_base_path}/BASE_FEATURES.pkl'):
        return pd.read_pickle(f'{save_base_path}/BASE_FEATURES.pkl')
    else:
        grid_df = extract_features(read_base_path, TARGET)
        grid_df.to_pickle(f'{save_base_path}/BASE_FEATURES.pkl')
        return grid_df

def load_rolling_features(grid_df, save_base_path, target, shift):
    if os.path.exists(f'{save_base_path}/ROLLING{shift}_FEATURES.pkl'):
        return pd.read_pickle(f'{save_base_path}/ROLLING{shift}_FEATURES.pkl')
    else:
        grid_df = gen_rolling_features(grid_df, shift_day=shift, target=target)
        grid_df.to_pickle(f'{save_base_path}/ROLLING{shift}_FEATURES.pkl')
        return grid_df


########################### Apply on grid_df
#################################################################################
# lets read grid from
# https://www.kaggle.com/kyakovlev/m5-simple-fe
# to be sure that our grids are aligned by index
def gen_rolling_features(grid_df, shift_day=28, target='sales', verbose=0):

    # We need only 'id','d','sales'
    # to make lags and rollings
    # grid_df = pd.read_pickle(f'{base_path}/grid_part_1.pkl')
    grid_df = grid_df[['id', 'd', 'sales']]

    # Lags
    # with 28 day shift
    start_time = time.time()
    if verbose > 0:
        print('Create lags')
    LAG_DAYS = [col for col in range(shift_day, shift_day + 15)]
    grid_df = grid_df.assign(**{
        '{}_lag_{}'.format(col, l): grid_df.groupby(['id'])[col].transform(lambda x: x.shift(l))
        for l in LAG_DAYS
        for col in [target]
    })

    # Minify lag columns
    for col in list(grid_df):
        if 'lag' in col:
            grid_df[col] = grid_df[col].astype(np.float16)
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
        grid_df['rolling_mean_' + str(i)] = grid_df.groupby(['id'])[target].transform(
            lambda x: x.shift(shift_day).rolling(i).mean()).astype(np.float16)
        grid_df['rolling_std_' + str(i)] = grid_df.groupby(['id'])[target].transform(
            lambda x: x.shift(shift_day).rolling(i).std()).astype(np.float16)

    # Rollings
    # with sliding shift
    # grid_df = gen_sliding_shift_features(grid_df, TARGET)
    if verbose > 0:
        print('%0.2f min: Lags' % ((time.time() - start_time) / 60))

    ########################### Export
    #################################################################################
    if verbose > 0:
        print('Save lags and rollings')
    # grid_df.to_pickle(f'{base_path}/lags_df_{SHIFT_DAY}.pkl')
    return grid_df

def gen_sliding_shift_features(grid_df, target):
    # base_test = grid_df[['id','d',TARGET]]
    old_tmp_cols = [col for col in list(grid_df) if '_tmp_' not in col]
    grid_df = grid_df.drop(columns=old_tmp_cols)
    ROLS_SPLIT = []
    for i in [1, 7, 14]:
        for j in [7, 14, 30, 60]:
            ROLS_SPLIT.append([grid_df, target, i, j])
    grid_df = pd.concat([grid_df, _df_parallelize_run(_make_lag_roll, ROLS_SPLIT)], axis=1)
    return grid_df


def gen_encode_features(grid_df, TARGET, nan_mask_d = 1913-28, ):
    base_ = grid_df[['store_id', 'cat_id', 'dept_id', 'item_id', 'tm_dw', TARGET]].copy()
    base_[TARGET][grid_df['d'] > nan_mask_d] = np.nan
    icols = [
        ['store_id', 'cat_id'],
        ['store_id', 'dept_id'],
        ['store_id', 'item_id'],
        ['store_id', 'tm_dw', 'item_id'],
        ['store_id', 'tm_dw'],
    ]
    for col in icols:
        col_name = '_' + '_'.join(col) + '_'
        grid_df['enc' + col_name + 'mean'] = base_.groupby(col)[TARGET].transform('mean').astype(np.float16)
        grid_df['enc' + col_name + 'std'] = base_.groupby(col)[TARGET].transform('std').astype(np.float16)
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

def _make_lag_roll(base_test, TARGET, LAG_DAY):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    lag_df = base_test[['id','d',TARGET]]
    col_name = 'rolling_mean_tmp_'+str(shift_day)+'_'+str(roll_wind)
    lag_df[col_name] = lag_df.groupby(['id'])[TARGET].transform(lambda x: x.shift(shift_day).rolling(roll_wind).mean())
    return lag_df[[col_name]]


