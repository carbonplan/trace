import os

import numpy as np
import pandas as pd
import xgboost as xgb
from s3fs import S3FileSystem
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

fs = S3FileSystem()

features = (
    ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'NDVI', 'NDII', 'NIR_V']
    + [f'BIO{str(n).zfill(2)}' for n in range(1, 20)]
    + ['burned', 'treecover2000_mean']
    + ['prec', 'srad', 'tavg', 'tmax', 'tmin', 'vapr', 'wind']
    + ['elev', 'slope', 'aspect']
)
label = 'biomass'

RED_BAND = 'SR_B3'
NIR_BAND = 'SR_B4'
SWIR_BAND = 'SR_B5'


def calc_NDII(df):
    nir = df[NIR_BAND]
    swir = df[SWIR_BAND]
    return (nir - swir) / (nir + swir)


def calc_NDVI(df):
    nir = df[NIR_BAND]
    red = df[RED_BAND]
    return (nir - red) / (nir + red)


def calc_NIR_V(df):
    """
    https://advances.sciencemag.org/content/3/3/e1602244?intcmp=trendmd-adv
    """
    if 'NDVI' not in df:
        df['NDVI'] = calc_NDVI(df)

    return df[NIR_BAND] * df['NDVI']


def train_test_split_based_on_year(
    df, val_strategy='last', random_train_test=False, seed=0, test_size=0.2
):
    """
    split into train, test, and val data based different strategies

    val_year can either be 'first' or 'last', where either the first or last year of data will be used
    for out of sample validation

    if random_train_test then the rest of the data will be split randomly for train/test, else train/test
    will also be split by year
    """
    all_years = sorted(df.year.unique())
    if val_strategy == 'first':
        val_year = all_years[0]
        test_year = all_years[1]
    elif val_strategy == 'last':
        val_year = all_years[-1]
        test_year = all_years[-2]
    else:
        raise Exception('val_year has to be either first or last')

    df_val = df.loc[df.year == val_year].reset_index()

    if random_train_test:
        # use biomass value to do stratified sampling
        df_rest = df.loc[df.year != val_year].reset_index()
        kmeans = KMeans(n_clusters=10, random_state=seed).fit(df_rest[['biomass']].values)
        df_rest['biomass_cluster'] = kmeans.predict(df_rest[['biomass']].values)
        df_train, df_test = train_test_split(
            df_rest, test_size=test_size, random_state=seed, stratify=df_rest['biomass_cluster']
        )
    else:
        df_test = df.loc[df.year == test_year].reset_index(drop=True)
        df_train = df.loc[~df.year.isin([val_year, test_year])].reset_index(drop=True)

    return df_train, df_test, df_val


def fit_transformers(df, columns):
    transformers = {}
    for col in columns:
        transformers[col] = StandardScaler().fit(df[[col]])
    return transformers


def transform_df(transformers, df):
    out = pd.DataFrame(index=df.index)
    for key in df:
        if key in transformers:
            out[key] = transformers[key].transform(df[[key]])
        else:
            out[key] = df[key]
    return out


def get_features_and_label(df):
    df.loc[:, 'NIR_V'] = calc_NIR_V(df)

    for v in features + [label]:
        assert v in df

    return df[features], df[label]


def mean_error(y_true, y_pred):
    return np.mean(y_pred - y_true)


def load_xgb_model(model_path, fs):
    cwd = os.getcwd()
    if model_path.startswith('s3'):
        model_name = model_path.split('/')[-1]
        new_model_path = ('/').join([cwd, model_name])
        fs.get(model_path, new_model_path)
        model_path = new_model_path

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    return model


def save_xgb_model(model, model_path, fs):
    cwd = os.getcwd()
    if model_path.startswith('s3'):
        model_name = model_path.split('/')[-1]
        local_model_path = ('/').join([cwd, model_name])
        model.save_model(local_model_path)
        fs.put_file(lpath=local_model_path, rpath=model_path)
    else:
        model.save_model(model_path)


eval_funcs = {'bias': mean_error, 'mae': mean_absolute_error, 'r2': r2_score}


class baseline_model:
    """
    a model that always predicts the mean of training label
    """

    def __init__(
        self,
        realm,
        df_train,
        df_test,
        output_folder,
        validation_year,
        params=None,
        overwrite=False,
        seed=42,
    ):
        self.name = f'baseline_{realm}'
        self.eval_funcs = eval_funcs
        self.model_filename = f'{output_folder}{self.name}.txt'
        if fs.exists(self.model_filename) and not overwrite:
            print(f'    {self.name} model already exists, loading')
            self._load()
        else:
            print(f'    Building {self.name} model')
            self.X_train, self.y_train = get_features_and_label(df_train)
            self.X_test, self.y_test = get_features_and_label(df_test)
            self._fit()
            self._save()

    def _load(self):
        with fs.open(self.model_filename) as f:
            self.model = float(f.read())

    def _fit(self):
        self.model = np.mean(self.y_train)

    def _predict(self, df):
        return [self.model] * len(df)

    def _save(self):
        with fs.open(self.model_filename, 'w') as f:
            f.write(str(self.model))

    def evaluate(self, df):
        y_pred = self._predict(df)
        out = pd.DataFrame()
        out['y_true'] = df[label]
        out['y_pred'] = y_pred
        n = len(out)
        out = out.dropna(how='any')
        if len(out) < n:
            print(f'    dropping {n-len(out)} samples due to nans')
        scores = {}
        for k, func in self.eval_funcs.items():
            scores[k] = func(out.y_true.values, out.y_pred.values)

        return scores


class xgb_model(baseline_model):
    """
    XGBoost model
    """

    def __init__(
        self,
        realm,
        df_train,
        df_test,
        output_folder,
        validation_year,
        params={},
        overwrite=False,
        seed=42,
    ):
        self.name = f'xgb_{realm}_{validation_year}'
        self.eval_funcs = eval_funcs
        self.model_filename = f'{output_folder}{self.name}.bin'
        self.transformer_filename = f'{output_folder}{self.name}_transformers.pkl'
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 999,
            'random_state': seed,
            'learning_rate': 0.05,
            'max_depth': 10,
            'colsample_bytree': 0.7,
            'subsample': 0.7,
            'min_child_weight': 4,
            # 'lambda': 2,
            # 'alpha': 1,
            # 'gamma': 1,
        }
        base_params.update(params)
        self.params = base_params
        print(self.params)

        if fs.exists(self.model_filename) and not overwrite:
            print(f'    {self.name} model already exists, loading')
            self._load()
        else:
            print(f'    Building {self.name} model')

            # cols_to_scale = [
            #     'SR_B1',
            #     'SR_B2',
            #     'SR_B3',
            #     'SR_B4',
            #     'SR_B5',
            #     'SR_B7',
            #     'NDVI',
            #     'NDII',
            #     'aspect',
            #     'elev',
            #     'slope',
            #     'BIO01',
            #     'BIO02',
            #     'BIO03',
            #     'BIO04',
            #     'BIO05',
            #     'BIO06',
            #     'BIO07',
            #     'BIO08',
            #     'BIO09',
            #     'BIO10',
            #     'BIO11',
            #     'BIO12',
            #     'BIO13',
            #     'BIO14',
            #     'BIO15',
            #     'BIO16',
            #     'BIO17',
            #     'BIO18',
            #     'BIO19',
            #     'prec',
            #     'srad',
            #     'tavg',
            #     'tmax',
            #     'tmin',
            #     'vapr',
            #     'wind',
            # ]
            # self.transformers = fit_transformers(df_train, columns=cols_to_scale)
            # # save transformers
            # with fs.open(self.transformer_filename, 'wb') as f:
            #     pickle.dump(self.transformers, f)

            # df_train_scaled = transform_df(self.transformers, df_train)
            # df_test_scaled = transform_df(self.transformers, df_test)

            self.X_train, self.y_train = get_features_and_label(df_train)
            self.X_test, self.y_test = get_features_and_label(df_test)
            self._fit()
            self._save()

    def _load(self):
        self.model = load_xgb_model(self.model_filename, fs)
        # # load transformers
        # with fs.open(self.transformer_filename, 'rb') as f:
        #     self.transformers = pickle.load(f)

    def _fit(self):
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            early_stopping_rounds=10,
        )

    def _predict(self, df):
        try:
            ntree_limit = self.model.best_ntree_limit
        except AttributeError:
            ntree_limit = None

        # df_scaled = transform_df(self.transformers, df)
        X, y = get_features_and_label(df)
        if ntree_limit:
            return self.model.predict(X, iteration_range=(0, ntree_limit))
        else:
            return self.model.predict(X)

    def _save(self):
        save_xgb_model(self.model, self.model_filename, fs)


# def linear_model(X_train, X_test, y_train, y_test):
#     model = SGDRegressor()
#     model.fit(X_train, y_train)
#     return model

# def random_forest_model(X_train, X_test, y_train, y_test, seed=0):
#     model = RandomForestRegressor(n_estimators=200, max_features='auto', max_depth=None, min_samples_leaf=1, random_state=seed)
#     model.fit(X_train, y_train)
#     return model
