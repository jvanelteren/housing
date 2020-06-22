
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import TransformerMixin,BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
class DFSimpleImputer(SimpleImputer):
    def transform(self, X):
        return pd.DataFrame(super().transform(X),columns=X.columns)

class DFOneHotEncoder(OneHotEncoder):
    def transform(self, X):
        # return super().transform(X)
        arr = super().transform(X)
        # print('OHE',arr.shape, self.get_feature_names().shape)
        return pd.DataFrame.sparse.from_spmatrix(arr,columns=self.get_feature_names())
        # return arr, self.get_feature_names()

class DFMinMaxScaler(MinMaxScaler):
    def transform(self, X):
        return pd.DataFrame(super().transform(X),columns=X.columns)
class DFColumnTransformer(ColumnTransformer):
    # works only with non-sparse matrices!
    def _hstack(self, Xs):
        Xs = [f for f in Xs]
        cols = [col for f in Xs for col in f.columns]
        df = pd.DataFrame(np.hstack(Xs), columns=cols)
        # print('final shape',df.shape)
        return df.infer_objects()

class DFOutlierExtractor(BaseEstimator, RegressorMixin):

    def __init__(self, model, **kwargs):
        """ 
        Keyword Args:
        neg_conf_val (float): The threshold for excluding samples with a lower
        negative outlier factor.
        """
    
        self.model = model
        self.threshold = kwargs.pop('neg_conf_val', -10.0)
        self.kwargs = kwargs
        self.lcf = []


    def fit(self, X, y):
        xs = np.asarray(X)
        ys = np.asarray(y)
        lcf = LocalOutlierFactor(**self.kwargs)
        lcf.fit(xs)
        self.xs  = pd.DataFrame(xs[lcf.negative_outlier_factor_ > self.threshold, :],columns=X.columns)
        self.ys = y[lcf.negative_outlier_factor_ > self.threshold]
        self.lcf  = lcf
        print(xs.shape,ys.shape,X.shape)
        self.model.fit(xs,ys)
        return self


    def predict(self, X):
        return self.model.predict(X)