from skopt.space import Real, Categorical, Integer
from dataclasses import dataclass,field
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet,SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

class AutoCatBoostRegressor(CatBoostRegressor):
    def fit(self,X,y,**kwargs):
        categorical = list(X.dtypes[(X.dtypes=='category' )| (X.dtypes=='object') | (X.dtypes=='O')].index)
        # print(X.shape,len(categorical),categorical)
        res =  super().fit(X,y,cat_features=categorical)
        return res
        

@dataclass
class Param:
     model: object = None
     hyper: dict = None
     preprocess: dict = field(default_factory=lambda: {})
     
models = {    
            'LinearRegression':Param(LinearRegression(),
                {}),

            'Lasso':Param(Lasso(),
                {'alpha':(0.00001,1.0,'log-uniform')}),

            'Ridge': Param(Ridge(),
                {'alpha':(0.00001,1.0,'log-uniform')}),

            'ElasticNet':Param(ElasticNet(),
                {'l1_ratio': Real(0.01, 1.0, 'log-uniform'),
                'alpha':Real(0.0001, 1.0, 'log-uniform')
                }),

            'svm.SVR': Param(svm.SVR(), 
                # https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html#advanced-example
                # https://scikit-learn.org/stable/modules/svm.html#regression
                {'C': Real(0.1, 20, 'log-uniform'),
                'degree': Integer(1, 8),  # integer valued parameter
                'epsilon': Real(0.004, 0.01),  # integer valued parameter
                'gamma': Real(0, 0.001),  # integer valued parameter
                'kernel': ['linear', 'rbf']}),

            'KNeighborsRegressor':Param(KNeighborsRegressor(),
                {'n_neighbors': (2,3,4,5,6), 
                'weights': ['uniform','distance']}),

            'RandomForestRegressor':Param(RandomForestRegressor(),
            #
            #fastai:
            #One of the most important properties of random forests is that they aren't very sensitive to the hyperparameter choices, such as max_features. You can set n_estimators to as high a number as you have time to train—the more trees you have, the more accurate the model will be. max_samples can often be left at its default, unless you have over 200,000 data points, in which case setting it to 200,000 will make it train faster with little impact on accuracy. max_features=0.5 and min_samples_leaf=4 both tend to work well, although sklearn's defaults work well too.
                {'n_estimators' : (1,100, 'log-uniform'), # gamble
                'max_depth' : (3, 40,'log-uniform'),
                'max_features':Real(0.4,0.7,'uniform'),
                'min_samples_leaf': Integer(3,8,'uniform')}),

            # SGD does not work atm with bayessearch (not hashable)
            # 'SGDRegressor':Param(SGDRegressor(),
            #     {'loss' : ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
            #     'penalty' : ['l1', 'l2', 'elasticnet'],
            #     'alpha' : (0.0,1000,'log-uniform'),
            #     'learning_rate' : ['constant', 'optimal', 'invscaling', 'adaptive'],
            #     'class_weight' : [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
            #     'eta0' : [1, 10, 100]}),

            # Catboost does not work atm with bayessearch. Tried to define __hash__, but then got into problems with is_comparable
            'AutoCatBoostRegressor':Param(AutoCatBoostRegressor(silent=True,one_hot_max_size=20, iterations = 300),
                {
                # 'iterations': Integer(350,350),
                 'depth': Integer(4, 10),
                 'learning_rate': Real(0.01, 1.0, 'log-uniform'),
                 'random_strength': Real(1e-9, 10, 'log-uniform'),
                 'bagging_temperature': Real(0.0, 1.0),
                 'border_count': Integer(1, 255),
                 'l2_leaf_reg': Integer(2, 30),
                #  'scale_pos_weight':Real(0.01, 1.0, 'uniform')
                 },
            preprocess={'onehot':False}),

            'xgb.XGBRegressor':Param(xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, random_state =7, nthread = -1, silent=True),
                {'max_depth': Integer(2, 10,'log-uniform'), #9,12
                'min_child_weight': Integer(0, 4,'uniform'), # if leaves with small amount of observations are allowed?
                'gamma' : Real(0,0.1), # these 3 for model complexity. gemma is a threshold for gain of the new split. 
                # 'subsample': (1,),
                'colsample_bytree' : Real(0.2,1.0,'log-uniform'), # these 3 for making model more robust to noise
                'reg_lambda' : Real(0.0,0.9,'uniform'), # regularization lambda, reduces similarity scores and therefore lowers gain. reduces sensitivity to individual observations
                'colsample_bylevel': Real(0.7,1.0,'log-uniform'),
                'learning_rate': Real(0.001,0.4,'log-uniform'), # 0.3 is default
                'max_delta_step': Real(0.0,10.0,'uniform'),
                'n_estimators': Integer(10,2500,'log-uniform')}), # number of trees to build
            #https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/89994

            'lgb.LGBMRegressor': Param(lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=500,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11),
                        # num_leaves=31,
                        # learning_rate=0.05,
                        # n_estimators=20,
                {'num_leaves': Integer(3, 8),
                'n_estimators':Integer(100, 1000),
                'min_data_in_leaf': Integer(3, 30),
                'max_depth': Integer(3, 12),
                'learning_rate': Real(0.01, 0.3,'log_uniform'), #'log_uniform'
                'bagging_freq': Integer(3, 7),
                'bagging_fraction': Real(0.6, 0.95, 'log_uniform'),
                'reg_alpha': Real(0.1, 0.95, 'log_uniform'),
                'reg_lambda': Real(0.1, 0.95, 'log_uniform')
            }),

            # A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array
            'HistGradientBoostingRegressor':Param(HistGradientBoostingRegressor(loss='huber'),
                {'loss':'huber'}),
            'GradientBoostingRegressor': Param(GradientBoostingRegressor(learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5),
                {
                'max_depth': Integer(3,12),
                'min_samples_leaf': Integer(5,20),
                }),

            'LassoCV':Param(LassoCV(),
                {}),
                
            'MLPRegressor':Param(MLPRegressor(),
                {})
                }