
from sklearn.experimental import enable_iterative_imputer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import TransformerMixin,BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn.impute import IterativeImputer
from sklearn.compose import make_column_selector
from tempfile import mkdtemp
# todo multivariate imputation, possibly with pipelines for numeric and categorical data

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
# https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
# https://stackoverflow.com/questions/51237635/difference-between-standard-scaler-and-minmaxscaler/51237727
# don't know features are normal so just going with minmax scalar atm

from sklearn.preprocessing import OneHotEncoder
#https://stackoverflow.com/questions/36631163/what-are-the-pros-and-cons-between-get-dummies-pandas-and-onehotencoder-sciki
#The crux of it is that the sklearn encoder creates a function which persists and can then be applied to new data sets which use the same categorical variables, with consistent results.
# So don't use pandas get dummies, but a OneHotEncoder

from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
class DFSimpleImputer(SimpleImputer):
    def transform(self, X,y=None):
        return pd.DataFrame(super().transform(X),columns=X.columns)

class DFSmartImputer(TransformerMixin):
    def fit(self,X, y=None):
        self.conversion = {name: SimpleImputer(strategy='most_frequent').fit(df) for name, df in X.groupby('Neighborhood')}
        return self
    
    def transform(self, X,y=None):
        self.trans = [df.fillna(value=dict(zip(X.columns,self.conversion[name].statistics_))) for name, df in X.groupby('Neighborhood')]
        return pd.concat(self.trans)


class DFGetDummies(TransformerMixin):
    def fit(self, X, y=None):
        self.train = pd.get_dummies(X)
        return self
    def transform(self, X, y=None):
        self.test = pd.get_dummies(X)
        return self.test.reindex(columns=self.train.columns,fill_value=0)

class DFOneHotEncoder(OneHotEncoder):
    def transform(self, X,y=None):
        # return super().transform(X)
        arr = super().transform(X)
        # print('OHE',arr.shape, self.get_feature_names().shape)
        return pd.DataFrame.sparse.from_spmatrix(arr,columns=self.get_feature_names())
        # return arr, self.get_feature_names()

class DFMinMaxScaler(MinMaxScaler):
    def transform(self, X, y=None):
        return pd.DataFrame(super().transform(X),columns=X.columns)

class DFStandardScaler(StandardScaler):
    def transform(self, X, y=None):
        return pd.DataFrame(super().transform(X),columns=X.columns)
class DFRobustScaler(RobustScaler):
    def transform(self, X, y=None):
        return pd.DataFrame(super().transform(X),columns=X.columns)

class DFColumnTransformer(ColumnTransformer):
    # works only with non-sparse matrices!
    def _hstack(self, Xs):
        Xs = [f for f in Xs]
        cols = [col for f in Xs for col in f.columns]
        df = pd.DataFrame(np.hstack(Xs), columns=cols)
        # print('final shape',df.shape)
        return df.infer_objects()

class DFOutlierExtractor(TransformerMixin):

    def __init__(self, model, thres):
        """ 
        Keyword Args:
        neg_conf_val (float): The threshold for excluding samples with a lower
        negative outlier factor.
        """
        self.model = model
        self.threshold = thres

    def fit(self, X, y):
        xs = np.asarray(X)
        ys = np.asarray(y)
        lcf = LocalOutlierFactor()
        lcf = lcf.fit(xs)
        xs  = pd.DataFrame(xs[lcf.negative_outlier_factor_ > self.threshold, :],columns=X.columns)
        ys = y[lcf.negative_outlier_factor_ > self.threshold]
        print('removed',len(X) - len(xs),self.threshold)
        self.model.fit(xs,ys)
        return self

    def predict(self, X):
        return self.model.predict(X)



from joblib import Memory
cachedir = mkdtemp()
memory = Memory(cachedir, verbose=0)

def densify(x): # needs to use a function, lambda gives problems with pickling
    return x.todense()

def get_pipeline(model, scale=True,onehot=True,to_dense=False,remove_outliers=False, smart_imp=False):

    cat_steps = []
    if smart_imp: 
        cat_steps.append(('impute_cat', DFSmartImputer()))
    else:
        cat_steps.append(('impute_cat', DFSimpleImputer(strategy='most_frequent',fill_value='NaN')))
    if onehot: 
        cat_steps.append(('cat_to_num', DFOneHotEncoder(handle_unknown="ignore")))
    else:
        cat_steps.append(('cat_to_num', DFGetDummies()))
    categorical_transformer = Pipeline(steps=cat_steps)

    num_steps = []
    if False: 
        num_steps.append(('impute_num', DFSmartImputer()))
    else:
        num_steps.append(('impute_num', DFSimpleImputer(strategy='mean')))
    if scale: num_steps.append(('scale_num', DFMinMaxScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    col_trans = DFColumnTransformer(transformers=[
        ('numeric', numeric_transformer, make_column_selector(dtype_include=np.number)),
        ('category', categorical_transformer, make_column_selector(dtype_exclude=np.number)),
        ])
    
    preprocessor_steps = [('col_trans', col_trans)]
    preprocessor = Pipeline(steps=preprocessor_steps)

    final_pipe = [('preprocess', preprocessor)]
    if to_dense: final_pipe.append(('to_dense',FunctionTransformer(densify, accept_sparse=True)))
    if remove_outliers: 
        final_pipe.append(('model',DFOutlierExtractor(model, thres=-1.5)))
    else:
        final_pipe.append(('model',model))

    return Pipeline(steps=final_pipe)
    