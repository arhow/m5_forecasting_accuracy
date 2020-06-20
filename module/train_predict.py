import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from module.prepare_data import *

ORIGINAL = '../input/m5-forecasting-accuracy/'
STORES_IDS = pd.read_csv(ORIGINAL+'sales_train_validation.csv')['store_id']
STORES_IDS = list(STORES_IDS.unique())
TARGETS = 'sales'
BASE_PATH = '../cache/ver2'
START_TRAIN = 0
END_TRAIN = 1913
P_HORIZON = 28
SEED = 42
VER = 2
LGB_PARAMS = {
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    # 'tweedie_variance_power': 1.1,
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
                    'verbose': -1,
                }

CAT_COLUMNS = [
    'item_id',
 'dept_id',
 'cat_id',
 'event_name_1',
 'event_type_1',
 'event_name_2',
 'event_type_2',
 'snap_CA',
 'snap_TX',
 'snap_WI'
]

NUM_COLUMNS = [
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
 'tm_d',
 'tm_w',
 'tm_m',
 'tm_y',
 'tm_wm',
 'tm_dw',
 'tm_w_end',
 'enc_store_id_cat_id_mean',
 'enc_store_id_cat_id_std',
 'enc_store_id_dept_id_mean',
 'enc_store_id_dept_id_std',
 'enc_store_id_item_id_mean',
 'enc_store_id_item_id_std',
 'enc_store_id_tm_dw_item_id_mean',
 'enc_store_id_tm_dw_item_id_std',
 'enc_store_id_tm_dw_mean',
 'enc_store_id_tm_dw_std',
 'rolling_mean_7',
 'rolling_std_7',
 'rolling_mean_14',
 'rolling_std_14',
 'rolling_mean_30',
 'rolling_std_30',
 'rolling_mean_60',
 'rolling_std_60',
 'rolling_mean_180',
 'rolling_std_180']

########################### set seed
#################################################################################
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)

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

########################### Train Models
#################################################################################
def train_model(grid_df, features_columns, categorical_features, target, shift, save_base_path):
    for store_id in STORES_IDS:
        print('Train', store_id)
        train_mask = (grid_df['d'] > START_TRAIN) & (grid_df['d'] <= (END_TRAIN - P_HORIZON))
        valid_mask = (grid_df['d'] > (END_TRAIN - P_HORIZON - 200)) & (grid_df['d'] <= (END_TRAIN))
        train_data = lgb.Dataset(grid_df[train_mask][features_columns + categorical_features], label=grid_df[train_mask][target])
        valid_data = lgb.Dataset(grid_df[valid_mask][features_columns + categorical_features], label=grid_df[valid_mask][target])
        seed_everything(SEED)
        estimator = lgb.train(LGB_PARAMS, train_data, valid_sets=[valid_data], verbose_eval=100, categorical_feature=categorical_features)
        model_name = f'{save_base_path}/lgb_model_{store_id}_shift{shift}_v{VER}.bin'
        pickle.dump(estimator, open(model_name, 'wb'))
    return


def predict_samples(grid_df, base_path, store_id, shift_day, features_columns, categorical_features, target, verbose=1):
    ########################### Validation
    #################################################################################
    model_path = f'{base_path}/lgb_model_{store_id}_shift{shift_day}_v{VER}.bin'
    estimator = pickle.load(open(model_path, 'rb'))
    y_pred = estimator.predict(grid_df[features_columns])
    if type(target) == type(None):
        rmse_score = root_mean_sqared_error(grid_df['target'].values, y_pred)
    else:
        rmse_score = None
    return y_pred, rmse_score


    # load model and data

    # Create Dummy DataFrame to store predictions
    all_preds = pd.DataFrame()

    his = []
    # Join back the Test dataset with
    # a small part of the training data
    # to make recursive features
    base_test = get_base_test(base_path, typ)

    # Loop over each prediction day
    # As rolling lags are the most timeconsuming
    # we will calculate it for whole day
    for PREDICT_DAY in range(1, 29):
        PREDICT_DAY = shift_day
        if verbose > 0:
            print('Predict | Day:', PREDICT_DAY)
        start_time = time.time()

        # Make temporary grid to calculate rolling lags
        grid_df = base_test.copy()
        grid_df = pd.concat([grid_df, df_parallelize_run(make_lag_roll, ROLS_SPLIT)], axis=1)

        for store_id in STORES_IDS:
            # Read all our models and make predictions
            # for each day/store pairs
            model_path = f'{base_path}/lgb_model_{store_id}_v{VER}.bin'
            estimator = pickle.load(open(model_path, 'rb'))

            day_mask = base_test['d'] == (END_TRAIN - P_HORIZON + PREDICT_DAY)
            store_mask = base_test['store_id'] == store_id
            mask = (day_mask) & (store_mask)

            y_pred = estimator.predict(grid_df[mask][model_features])
            his.append({'store_id': store_id, 'd': END_TRAIN - P_HORIZON + PREDICT_DAY, 'y_pred': y_pred,
                        'y_true': base_test[target][mask].values})

        # Make good column naming and add
        # to all_preds DataFrame
        temp_df = base_test[day_mask][['id', target]]
        temp_df.columns = ['id', 'F' + str(PREDICT_DAY)]
        if 'id' in list(all_preds):
            all_preds = all_preds.merge(temp_df, on=['id'], how='left')
        else:
            all_preds = temp_df.copy()

        if verbose > 0:
            print('#' * 10, ' %0.2f min round |' % ((time.time() - start_time) / 60),
                  ' %0.2f min total |' % ((time.time() - main_time) / 60),
                  ' %0.2f day sales |' % (temp_df['F' + str(PREDICT_DAY)].sum()))
        del temp_df
        break

    all_preds = all_preds.reset_index(drop=True)
    return all_preds, pd.DataFrame(his)


BASE_GRID_DF = load_base_features(BASE_PATH, TARGETS)
print(BASE_GRID_DF.info())

for target in ['sales']:

    for model in ['lgb']:

        for store_id in STORES_IDS:

            for shift in range(1,29):

                print(f'{model} train {store_id} {shift} {target}')

                rolling_features_df = load_rolling_features(BASE_GRID_DF, BASE_PATH, TARGET='sales', SHIFT_DAY=shift)

                grid_df = pd.concat([BASE_GRID_DF, rolling_features_df.iloc[:, 2:]], axis=1)

                train_model(grid_df, features_columns, categorical_features, target, shift, BASE_PATH)
















