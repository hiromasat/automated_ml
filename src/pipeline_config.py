# set pipelines for different algorithms
# 学習手法
import numpy as np
import os
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

pipelines = {
    'knn':
        Pipeline([('scl',StandardScaler()),
            ('est',KNeighborsClassifier())]),
    'logistic':
        Pipeline([('scl',StandardScaler()),
            ('est',LogisticRegression(random_state=1))]),
    'rsvc':
        Pipeline([('scl',StandardScaler()),
            ('est',SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=1))]),
    'lsvc':
        Pipeline([('scl',StandardScaler()),
            ('est',LinearSVC(C=1.0, class_weight='balanced', random_state=1))]),
    'tree':
        Pipeline([('scl',StandardScaler()),
            ('est',DecisionTreeClassifier(random_state=1))]),
    'rf':
        Pipeline([('scl',StandardScaler()),
            ('est',RandomForestClassifier(random_state=1))]),
    'gb':
        Pipeline([('scl',StandardScaler()),
            ('est',GradientBoostingClassifier(random_state=1))]),
    'mlp':
        Pipeline([('scl',StandardScaler()),
            ('est',MLPClassifier(hidden_layer_sizes=(3,3),
                                    max_iter=1000,
                                    random_state=1))]),
    'xgb':
        Pipeline([('scl',StandardScaler()),
            ('est',xgb.XGBClassifier())]),
    'lgbm':
        Pipeline([('scl',StandardScaler()),
            ('est',lgb.LGBMClassifier())]),
}

# GridSearch parameters
gs_params = {
    'knn' : {'est__n_neighbors':[5,7,10],
            'est__weights':['uniform','distance'],},

    'logistic': {'est__C':[1, 100],},

    'rsvc': {'est__C':[1, 100],},

    'lsvc': {'est__C':[1, 100],},

    'tree': {'est__max_depth': list(range(10, 20)),
            'est__criterion': ['gini', 'entropy'],},

    'rf': {'est__n_estimators':[320, 340],
            'est__max_depth': [8, 10,],
            'est__random_state': [0],},

    'gb': {'est__loss':['deviance'],
            'est__learning_rate': [0.01, 0.1],
            'est__min_samples_split': np.linspace(0.1, 0.5, 2),
            'est__min_samples_leaf': np.linspace(0.1, 0.5, 2),
            'est__max_depth':[3,5],
            'est__max_features':['log2','sqrt'],
            'est__criterion': ['friedman_mse',  'mae'],
            'est__subsample':[0.5, 1.0],
            'est__n_estimators':[10],},

    'mlp': {'est__solver': ['lbfgs'],
            'est__max_iter': [10000],
            'est__alpha': 10.0 ** -np.arange(1, 3),
            'est__hidden_layer_sizes':np.arange(10, 12),},

    'xgb': {'est__n_estimators':[100,500,],
            'est__max_depth':[6, 8,10],
            'est__learning_rate':[0.001, 0.01, 0.1, 1],
            'est__min_child_weight': [1,6],},

    'lgbm': {'est__learning_rate':[0.001, 0.01,0.1,1],
            'est__n_estimators':[100, 500,],},
}


def fit_on_pipelines(pipelines, gs_params, X_train, X_test, y_train, y_test, evaluation_scoring):
    
    # スコア格納用のscores初期化（Dict型）
    scores = {}

    # パイプラインの先頭にある文字列（例えば、'KNN')が pipe_name に、
    # 各パイプラインのインスタンスがpipelineに順次入る
    for pipe_name, pipeline in pipelines.items():
        print(pipe_name)
        gs = GridSearchCV(estimator=pipeline,
                        param_grid = gs_params[pipe_name],
                        scoring=evaluation_scoring,
                        cv=5,
                        return_train_score=False)
        # 学習
        gs.fit(X_train, y_train)
        scores[(pipe_name,'train')] = accuracy_score(y_train, gs.predict(X_train))
        scores[(pipe_name,'test')] = accuracy_score(y_test, gs.predict(X_test))
    
        # 各モデル格納用のディレクトリ を作成
        os.makedirs('../models/pipeline_models', exist_ok=True)
        # 各モデル保存(modelフォルダー)
        file_name = '../models/pipeline_models/'+pipe_name+'.pkl'
        pickle.dump(pipeline, open(file_name, 'wb'))

    return scores
