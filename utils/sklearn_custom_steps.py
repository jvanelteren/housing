
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import TransformerMixin
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

class OutlierExtractor(TransformerMixin):
    def __init__(self, **kwargs):
        """
        Create a transformer to remove outliers. A threshold is set for selection
        criteria, and further arguments are passed to the LocalOutlierFactor class

        Keyword Args:
            neg_conf_val (float): The threshold for excluding samples with a lower
               negative outlier factor.

        Returns:
            object: to be used as a transformer method as part of Pipeline()
        """

        self.threshold = kwargs.pop('neg_conf_val', -10.0)

        self.kwargs = kwargs

    def transform(self, X, y):
        """
        Uses LocalOutlierFactor class to subselect data based on some threshold

        Returns:
            ndarray: subsampled data

        Notes:
            X should be of shape (n_samples, n_features)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        lcf = LocalOutlierFactor(**self.kwargs)
        lcf.fit(X)
        return (X[lcf.negative_outlier_factor_ > self.threshold, :],
                y[lcf.negative_outlier_factor_ > self.threshold])

    def fit(self, *args, **kwargs):
        return self

