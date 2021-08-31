import os

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import dump, load
from s3fs import S3FileSystem
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

fs = S3FileSystem()

features = (
    ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'NDVI', 'NDII', 'NIR_V']
    + [f'BIO{str(n).zfill(2)}' for n in range(1, 20)]
    + ['treecover2000_mean']
    + ['prec', 'srad', 'tavg', 'tmax', 'tmin', 'vapr', 'wind']
    + ['elev', 'slope', 'aspect']
)

label = 'biomass'


def calc_NDII(df):
    nir = df['SR_B4']
    swir = df['SR_B5']
    return (nir - swir) / (nir + swir)


def calc_NDVI(df):
    nir = df['SR_B4']
    red = df['SR_B3']
    return (nir - red) / (nir + red)


def calc_NIR_V(df):
    """
    https://advances.sciencemag.org/content/3/3/e1602244?intcmp=trendmd-adv
    """
    if 'NDVI' not in df:
        df['NDVI'] = calc_NDVI(df)

    return df['SR_B4'] * df['NDVI']


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
    elif val_strategy == 'none':
        # when no validation year is entered, the split strategy has to be random since otherwise
        # we do not know which year should be used for test
        assert random_train_test
        val_year = None
        test_year = None
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
    if 'NIR_V' not in df:
        df.loc[:, 'NIR_V'] = calc_NIR_V(df)

    for v in features + [label]:
        assert v in df

    return df[features], df[label]


def get_features(df):
    if 'NIR_V' not in df:
        df.loc[:, 'NIR_V'] = calc_NIR_V(df)

    for v in features:
        assert v in df

    return df[features]


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


def load_sklearn_model(model_path, fs):
    cwd = os.getcwd()
    if model_path.startswith('s3'):
        model_name = model_path.split('/')[-1]
        new_model_path = ('/').join([cwd, model_name])
        fs.get(model_path, new_model_path)
        model_path = new_model_path

    model = load(model_path)

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


def save_sklearn_model(model, model_path, fs):
    cwd = os.getcwd()
    if model_path.startswith('s3'):
        model_name = model_path.split('/')[-1]
        local_model_path = ('/').join([cwd, model_name])
        dump(model, local_model_path)
        fs.put_file(lpath=local_model_path, rpath=model_path)
    else:
        dump(model, model_path)


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
        if validation_year == 'none':
            validation_year = 'all'
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

    def predict(self, df):
        return [self.model] * len(df)

    def _save(self):
        with fs.open(self.model_filename, 'w') as f:
            f.write(str(self.model))

    def evaluate(self, df):
        y_pred = self.predict(df)
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
        if validation_year == 'none':
            validation_year = 'all'
        self.name = f'xgb_{realm}_{validation_year}'
        self.eval_funcs = eval_funcs
        self.model_filename = f'{output_folder}{self.name}.bin'
        self.transformer_filename = f'{output_folder}{self.name}_transformers.pkl'
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'n_estimators': 999,
            'random_state': seed,
            'learning_rate': 0.2,
            'max_depth': 14,
            'colsample_bytree': 0.5,
            'subsample': 0.5,
            'min_child_weight': 6,
            'lambda': 3,
            'alpha': 2,
            'gamma': 2,
        }
        base_params.update(params)
        self.params = base_params
        print(self.params)

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
        self.model = load_xgb_model(self.model_filename, fs)

    def _fit(self):
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            early_stopping_rounds=10,
        )

    def predict(self, df):
        try:
            ntree_limit = self.model.best_ntree_limit
        except AttributeError:
            ntree_limit = None

        X = get_features(df)
        if ntree_limit:
            return self.model.predict(X, iteration_range=(0, ntree_limit))
        else:
            return self.model.predict(X)

    def _save(self):
        save_xgb_model(self.model, self.model_filename, fs)


class random_forest_model(baseline_model):
    """
    Random Forest model
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
        if validation_year == 'none':
            validation_year = 'all'
        self.name = f'rf_{realm}_{validation_year}'
        self.eval_funcs = eval_funcs
        self.model_filename = f'{output_folder}{self.name}.joblib'
        self.seed = seed

        if fs.exists(self.model_filename) and not overwrite:
            print(f'    {self.name} model already exists, loading from {self.model_filename}')
            self._load()
        else:
            print(f'    Building {self.name} model')
            self.X_train, self.y_train = get_features_and_label(df_train)
            self.X_test, self.y_test = get_features_and_label(df_test)
            self._fit()
            self._save()

    def _load(self):
        self.model = load_sklearn_model(self.model_filename, fs)

    def _fit(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=14,
            min_samples_leaf=4,
            max_features=0.5,
            max_samples=0.5,
            random_state=self.seed,
            n_jobs=-1,
            verbose=2,
        )

        self.model.fit(
            self.X_train,
            self.y_train,
        )

    def _save(self):
        save_sklearn_model(self.model, self.model_filename, fs)

    def predict(self, df):
        X = get_features(df)
        return self.model.predict(X)


class gradient_boost_model(baseline_model):
    """
    Gradient Boosting model
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
        if validation_year == 'none':
            validation_year = 'all'
        self.name = f'gb_{realm}_{validation_year}'
        self.eval_funcs = eval_funcs
        self.model_filename = f'{output_folder}{self.name}.bin'
        self.seed = seed

        print(f'    Building {self.name} model')

        self.X_train, self.y_train = get_features_and_label(df_train)
        self.X_test, self.y_test = get_features_and_label(df_test)
        self._fit()

    def _fit(self):
        self.model = GradientBoostingRegressor(
            max_depth=10,
            min_samples_leaf=4,
            max_features=0.9,
            random_state=self.seed,
            n_estimators=200,
            learning_rate=0.05,
        )

        self.model.fit(
            self.X_train,
            self.y_train,
        )

    def predict(self, df):
        # df_scaled = transform_df(self.transformers, df)
        X, y = get_features_and_label(df)
        return self.model.predict(X)
