
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
class DFSimpleImputer(SimpleImputer):
    def transform(self, X):
        return pd.DataFrame(super().transform(X),columns=X.columns)

class DFOneHotEncoder(OneHotEncoder):
    def transform(self, X):
        # return super().transform(X)
        arr = super().transform(X)
        print('OHE',arr.shape, self.get_feature_names().shape)
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
        return df.infer_objects()