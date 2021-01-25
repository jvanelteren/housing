
from sklearn.experimental import enable_iterative_imputer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import TransformerMixin,BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler, PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.impute import IterativeImputer
from sklearn.compose import make_column_selector
from tempfile import mkdtemp
import random
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
# https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
# https://stackoverflow.com/questions/51237635/difference-between-standard-scaler-and-minmaxscaler/51237727
# don't know features are normal so just going with minmax scalar atm

from sklearn.preprocessing import OneHotEncoder
#https://stackoverflow.com/questions/36631163/what-are-the-pros-and-cons-between-get-dummies-pandas-and-onehotencoder-sciki
# The crux of it is that the sklearn encoder creates a function which persists and can then be applied to new data sets which use the same categorical variables, with consistent results.

from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
class DFSimpleImputer(SimpleImputer):
    # just like SimpleImputer, but retuns a df
    # this approach creates problems with the add_indicator=True, since more columns are returned
    # so don't set add_indicator to True
    def transform(self, X,y=None):
        return pd.DataFrame(super().transform(X),columns=X.columns) 
    def __repr__(self):
        return f'SimpleImputer'

class DFZeroOrMeanImputer(TransformerMixin): 
    # only for numeric features, was not successful
    # my try to set nan values to zero if a certain fraction is 0
    # if not than impute with mean
    def fit(self, X, y=None):
        def determine_zero(series):
            perc_zero =  sum(series==0) / series.count()
            if perc_zero > 0.5: #zero occurs often: 
                return True
            else: 
                return False
        self.zero_cols = set([c for c in X.columns if determine_zero(X[c])])
        self.other_cols = set(X.columns) - self.zero_cols
        if self.zero_cols:
            self.zero_imp = DFSimpleImputer(strategy='constant', fill_value=0)
            self.zero_imp.fit(X[self.zero_cols])

        if self.other_cols:
            self.other_imp = DFSimpleImputer(strategy='mean')
            self.other_imp.fit(X[self.other_cols])
        return self
    
    def transform(self, X,y=None):
        assert set(X.columns) == self.zero_cols | self.other_cols
        zero_trans, other_trans = pd.DataFrame(),pd.DataFrame(),
        if self.zero_cols: zero_trans = self.zero_imp.transform(X[self.zero_cols])
        if self.other_cols: other_trans = self.other_imp.transform(X[self.other_cols])
        return pd.concat([other_trans,zero_trans], axis=1)
    def __repr__(self):
        return f'ZeroOrMeanImputer'

class DFSmartImputer(TransformerMixin): 
    # will impute based on category Neighborhood
    # this sounded smart, but didn't generate good scores
    def __init__(self,strategy='most_frequent'):
        super().__init__()
        self.strategy = strategy

    def remove_neigh(self,df):
        return df.drop(columns=['Neighborhood'])
    def fit(self,X, y=None):
        if all(X.dtypes=='category'): # categorical columns
            self.overall_imp = SimpleImputer(strategy=self.strategy).fit(X)
            self.specific_imps = {name: SimpleImputer(strategy=self.strategy).fit(df) for name, df in X.groupby('Neighborhood', sort=False)}
        else: # numerical columns
            self.overall_imp = SimpleImputer(strategy=self.strategy).fit(self.remove_neigh(X))
            self.specific_imps = {name: SimpleImputer(strategy=self.strategy).fit(self.remove_neigh(df)) for name, df in X.groupby('Neighborhood', sort=False)}
        return self
    
    def get_transform_dict(self, name, df, X):
        impute_values = [imp_val if imp_val==imp_val else self.overall_imp.statistics_[i] for i, imp_val in enumerate(self.specific_imps[name].statistics_)]
        return dict(zip(X.columns,impute_values))

    def transform(self, X,y=None):
        if all(X.dtypes=='category'): # categorical columns
            dfs = [df.fillna(value=self.get_transform_dict(name,df,X)) for name, df in X.groupby('Neighborhood')]
        else:
            dfs = [self.remove_neigh(df).fillna(value=self.get_transform_dict(name,df,X)) for name, df in X.groupby('Neighborhood')]
        return pd.concat(dfs)
    def __repr__(self):
        return f'DFSmartImputer'

class make_smart_column_selector():
    def __init__(self,dtype_include):
        self.dtype_include = dtype_include
    def __call__(self,train):
        return make_column_selector(dtype_include = self.dtype_include)(train)+['Neighborhood']

class DFGetDummies(TransformerMixin):
    # actually this should be identical to sklearn OneHotEncoder()
    def fit(self, X, y=None):
        self.train = pd.get_dummies(X)
        return self
    def transform(self, X, y=None):
        self.test = pd.get_dummies(X)
        return self.test.reindex(columns=self.train.columns,fill_value=0)
    def __repr__(self):
        return 'DFGetDummies'

class DFOneHotEncoder(OneHotEncoder):
    def transform(self, X,y=None):
        # return super().transform(X)
        arr = super().transform(X)
        # print('OHE',arr.shape, self.get_feature_names().shape)
        return pd.DataFrame.sparse.from_spmatrix(arr,columns=self.get_feature_names())
        # return arr, self.get_feature_names()
    def __repr__(self):
        return 'DFOneHotEncoder'

class DFMinMaxScaler(MinMaxScaler):
    def transform(self, X, y=None):
        return pd.DataFrame(super().transform(X),columns=X.columns)
    def __repr__(self):
        return 'DFMinMaxScaler'

class DFStandardScaler(StandardScaler):
    def transform(self, X, y=None):
        return pd.DataFrame(super().transform(X),columns=X.columns)
    def __repr__(self):
        return 'DFStandardScaler'

class DFRobustScaler(RobustScaler):
    def transform(self, X, y=None):
        return pd.DataFrame(super().transform(X),columns=X.columns)
    def __repr__(self):
        return 'DFRobustScaler'

class DFPowerTransformer(PowerTransformer):
    def transform(self, X, y=None):
        return pd.DataFrame(super().transform(X),columns=X.columns)
    def __repr__(self):
        return 'DFPowerTransformer'

class DFColumnTransformer(ColumnTransformer):
    # works only with non-sparse matrices!
    def _hstack(self, Xs):
        Xs = [f for f in Xs]
        cols = [col for f in Xs for col in f.columns]
        df = pd.DataFrame(np.hstack(Xs), columns=cols)
        # print('final shape',df.shape)
        return df.infer_objects()

class DFOutlierExtractor(TransformerMixin,BaseEstimator):
    # automatically removes data from the dataset. Screws up CV and was not very successful.
    def __init__(self, model, thres=-1.5, contamination=None, verbose=False, **kwargs):
        """ 
        Keyword Args:
        neg_conf_val (float): The threshold for excluding samples with a lower
        negative outlier factor.
        """
        self.model = model
        self.threshold = thres
        self.contamination = contamination
        self.verbose = verbose
        self._estimator_type = "regressor"
        if kwargs:
            for k,v in kwargs.items(): setattr(self.model,k,v)
    def set_params(self,**kwargs):
        if self.verbose: print('set params called',kwargs)
        self.model.set_params(**kwargs)

    def __repr__(self):
        return f'DFOutlierExtractor thres {self.threshold}, cont {self.contamination}'

    def fit(self, X, y):
        xs = np.asarray(X)
        ys = np.asarray(y)
        if self.contamination: 
            lcf = LocalOutlierFactor(contamination=self.contamination)
        else:
            lcf = LocalOutlierFactor()
        lcf = lcf.fit(xs)
        if self.contamination: self.threshold = lcf.offset_
        xs  = pd.DataFrame(xs[lcf.negative_outlier_factor_ > self.threshold, :],columns=X.columns)
        ys = y[lcf.negative_outlier_factor_ > self.threshold]
        if self.verbose: print('removed',len(X) - len(xs),self.threshold, 'thres',self.threshold)
        self.model.fit(xs,ys)
        return self

    def predict(self, X):
        return self.model.predict(X)


from joblib import Memory
cachedir = mkdtemp()
memory = Memory(cachedir, verbose=0)

def get_pipeline(model, impute_cat='default', impute_num = 'default', scale='default',onehot='default',remove_outliers='default'):
    # in essence this splits the input into a categorical pipeline and a numeric pipeline
    # merged with a ColumnTransformer
    # on top a model is plugged (within OutlierExtractor if remove_outliers = True)
    # this works very nicely!

    cat_steps = []
    if impute_cat=='default':
        cat_steps.append(('impute_cat', DFSimpleImputer(strategy='constant',fill_value='None')))
    elif impute_cat:
        cat_steps.append(('impute_cat', impute_cat))
    
    if onehot == 'default':
        cat_steps.append(('cat_to_num', DFGetDummies()))
    elif onehot: 
        cat_steps.append(('cat_to_num', onehot))
        # equal to: cat_steps.append(('cat_to_num', DFOneHotEncoder(handle_unknown="ignore")))
    categorical_transformer = Pipeline(steps=cat_steps)

    num_steps = []
    if impute_num == 'default':
        num_steps.append(('impute_num', DFSimpleImputer(strategy='mean')))
    elif impute_num:
        num_steps.append(('impute_num', impute_num))
    
    if scale == 'default': 
        num_steps.append(('scale_num', DFStandardScaler()))
    elif scale:
        num_steps.append(('scale_num', scale))

    numeric_transformer = Pipeline(steps=num_steps)

    col_trans = DFColumnTransformer(transformers=[
        ('numeric', numeric_transformer, make_column_selector(dtype_include=np.number)),
        ('category', categorical_transformer, make_column_selector(dtype_exclude=np.number)),
        ])
    
    preprocessor_steps = [('col_trans', col_trans)]
    preprocessor = Pipeline(steps=preprocessor_steps,memory=memory)

    final_pipe = [('preprocess', preprocessor)]
    if remove_outliers == 'default': 
        final_pipe.append(('model',model))
    elif remove_outliers:
        final_pipe.append(('model',remove_outliers)) # DFOutlierExtractor(model, corruption=0.005)

    return Pipeline(steps=final_pipe)
    