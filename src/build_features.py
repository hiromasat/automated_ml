
import pandas as pd


# 欠損値補完
from sklearn.impute import SimpleImputer

# 次元圧縮
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# one hot encoding(カテゴリカル変数=>ohe)
def one_hot_encoding(data, ohe_columns):
    X_ohe = pd.get_dummies(data,
                       dummy_na=True,    # 欠損値もダミー化
                       columns=ohe_columns)
    X_ohe_columns = X_ohe.columns.values
    print('X_ohe shape:(%i,%i)' % X_ohe.shape)
    return X_ohe, X_ohe_columns


# モデル用データの前処理：数値変数の欠損値対応
def imputing_nan(X_ohe_for_training, X_ohe_apply_to):
    
    imp = SimpleImputer()   # default設定で平均値
    imp.fit(X_ohe_for_training)     # impにて計算するデータ
    
    X_ohe_columns =  X_ohe_for_training.columns.values
    X_ohe = pd.DataFrame(imp.transform(X_ohe_apply_to), columns=X_ohe_columns)

    return X_ohe, imp

# 次元圧縮
def dimension_compression(X_ohe, y):
    #selector = RFE(RandomForestClassifier(n_estimators=100, random_state=1),
    #           n_features_to_select=15, # 圧縮後の次元数
    #           step=.05)
    selector = RFECV(estimator=RandomForestClassifier(n_estimators=100,random_state=0), step=0.05)
    selector.fit(X_ohe,y)
    X_ohe_columns =  X_ohe.columns.values
    # 学習用のデータセットを処理
    # selector.support_には、True/Falseのリストとなっている
    X_fin = X_ohe.loc[:, X_ohe_columns[selector.support_]]
    print('X_fin shape:(%i,%i)' % X_fin.shape)
    return X_fin, selector


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

def compare_train_test_data(X_ohe, X_ohe_s):
    cols_model= set(X_ohe.columns.values)
    cols_score = set(X_ohe_s.columns.values)

    diff1 = cols_model - cols_score
    print('モデル用データのみに存在する項目: %s' %diff1)

    diff2 = cols_score - cols_model
    print('スコア用データのみに存在する項目: %s' %diff2)

# one-hot-encoding後のデータ不整合の解消
# モデル用にはあるが、スコア用に存在しない変数は復活させる
# スコア用データにあるが、モデル用に存在しない変数は削除する
def resolution_for_inconsistent_data(X_ohe, X_ohe_s):
    dataset_cols_m = pd.DataFrame(None,    # 空のデータ
                     columns=X_ohe.columns.values,# モデリング時のone-hot-encoding後のカラム構成
                     dtype=float)
    X_ohe_s = pd.concat([dataset_cols_m, X_ohe_s])

    # スコアリングデータのみに登場する変数を削除する
    set_Xm = set(X_ohe.columns.values)
    set_Xs = set(X_ohe_s.columns.values)
    X_ohe_s = X_ohe_s.drop(list(set_Xs-set_Xm),axis=1)

    # スコアリングでは登場しなかったデータ項目をゼロで埋める
    X_ohe_s.loc[:,list(set_Xm-set_Xs)] = X_ohe_s.loc[:,list(set_Xm-set_Xs)].fillna(0,axis=1)

    return X_ohe_s
    
# データ項目の並び順担保
def reindex_data(X_ohe_columns, X_ohe_s):
    return X_ohe_s.reindex(X_ohe_columns, axis=1)

# 10-6. 欠損値処理
def transform_missing_value_tozero(X_ohe_s, imp):
    print('欠損個数（数値変数の欠損補完前）',X_ohe_s.isnull().sum().sum())    # rowをsum()して、columnをsum()する
    # (重要)モデリングデータで作ったimpを使ってtransformする
    # もしここで改めてimpしたら、改めて計算されてしまう。そのためモデリングデータで使った平均値データを使ってtransformする
    X_ohe_s = pd.DataFrame(imp.transform(X_ohe_s),columns=X_ohe_s.columns.values)
    print('欠損個数（数値変数の欠損補完後）',X_ohe_s.isnull().sum().sum())
    return X_ohe_s
