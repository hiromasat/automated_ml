'''
1. モジュールのインポート  
2. 定数の設定  
3. データセットの読み込み  
4. EDA(データを知る)  
    欠損値の確認  
    各データのタイプ確認  
5. 学習用データセットの作成  
    5-1. ohe-encoding  
    5-2. 欠損値補完  
    5-3. 次元圧縮、特徴量選択  
    (5-4. 不均衡データへの対応)  
6. Pipelineの設定  
    6-1. Pipelineの定義  
    6-2. Grid Search用のパラメータ定義  
7. 学習  
8. モデル選定  
9. 全データを使った学習  
10. スコアリング用のデータ作成  
    10-1. スコアリング用のデータの読み出し  
    10-2. one-hot-encoding処理  
    10-3. one-hot encoding後のデータ整合チェック  
    10-4. one-hot-encoding後のデータ不整合の解消  
    10-5. データ項目の並び順の担保  
    10-6. 欠損値処理  
11. 推論  
12. 提出用データ(CSVファイル）の作成  
'''

# データ解析のライブラリ
import numpy as np
import pandas as pd
import os
import pickle
import time

# Scikit-learn
from sklearn.model_selection import train_test_split

# build_features.py
from build_features import one_hot_encoding
from build_features import imputing_nan
from build_features import dimension_compression
from build_features import compare_train_test_data
from build_features import resolution_for_inconsistent_data
from build_features import reindex_data
from build_features import transform_missing_value_tozero

# load_data.py
from load_data import dtype_columns
from load_data import data_load

# config.py
import config

# pipeline.py
import pipeline_config
from pipeline_config import fit_on_pipelines

# functions.py
from functions import print_on_terminal
from functions import make_submission_file


# メイン関数
def main():

    # カテゴリ変数として指定したカラムのデータタイプを指定する
    my_dtype = dtype_columns(ohe_columns) 

    # 3. データセットの読み込み
    dataset, X, y = data_load(train_file, train_reject_items, my_dtype, target_value)

    # ターミナル上に表示
    print_on_terminal('# check the dataset shape', dataset, X, y)

    # 5-1.one-hot-encoding
    X_ohe, X_ohe_columns = one_hot_encoding(X, ohe_columns)

    # 5-2. 欠損値処理
    X_ohe, imp = imputing_nan(X_ohe, X_ohe)

    # 5-3. 次元圧縮
    X_fin, selector = dimension_compression(X_ohe, y)

    # ターミナル上に表示
    print_on_terminal('前処理後の訓練データ', dataset, X_fin, y)

    # 7. 学習
    # Holdout(訓練データとテストデータに分ける)
    X_train, X_test, y_train, y_test=train_test_split(X_fin,
                                                       y,
                                                       test_size=0.3,
                                                       random_state=1)

    # 評価指標
    evaluation_scoring = 'f1'

    # 学習の実行
    scores = fit_on_pipelines(pipelines, gs_params, X_train, X_test, y_train, y_test, evaluation_scoring)

    print(pd.Series(scores).unstack())

    # 8. モデルの選定
    final_model_name = 'lgbm'

    # 9. 全データを使った学習の実行    
    final_model = pipelines[final_model_name].fit(X_fin, y)
    print(final_model.score(X_fin, y))

    # モデル格納用のディレクトリ を作成
    os.makedirs('../models/final_model', exist_ok=True)
    # モデル保存(modelフォルダー)
    file_name = '../models/final_model/'+'final_model_'+ final_model_name +'.pkl'
    pickle.dump(final_model, open(file_name, 'wb'))

    # 10. スコアリングフェーズ
    # 10-1. データセットの読み込み
    dataset_s, X_s, _ = data_load(test_file, score_reject_items, my_dtype, target_value)

    # 形状の確認
    print('-----------------------------------')
    print('Raw Shape: (%i, %i)' %dataset_s.shape)
    print('X_s Shape: (%i, %i)' %X_s.shape)
    print('-----------------------------------')

    # 10-2. one-hot-encoding処理
    # X_ohe_s, X_ohe_s_columns = one_hot_encoding(X_s, ohe_columns)
    X_ohe_s, _ = one_hot_encoding(X_s, ohe_columns)

    # 10-3. one-hot encoding後のデータ整合チェック
    compare_train_test_data(X_ohe, X_ohe_s)

    # 10-4. one-hot-encoding後のデータ不整合の解消
    X_ohe_s = resolution_for_inconsistent_data(X_ohe, X_ohe_s)

    # 10-5. データ項目の並び順担保
    X_ohe_s = reindex_data(X_ohe_columns, X_ohe_s)

    # 10-6. 欠損値処理
    X_ohe_s = transform_missing_value_tozero(X_ohe_s, imp)

    # 10-7. 次元圧縮、特徴量選択
    X_fin_s = X_ohe_s.loc[:, X_ohe_columns[selector.support_]]
    print(X_fin_s.shape)
    print('-----------------------------------')
    print('X_fin_s shape: (%i,%i)' %X_fin_s.shape)
    print('-----------------------------------')

    # 11. 推論
    y_pred = final_model.predict(X_fin_s)

    # 12. 提出用データ(CSVファイル)の作成
    make_submission_file(ID_column, dataset_s, target_value, y_pred, submit_file_dir, submit_file_name)


if __name__ == '__main__':
    # SET PARAMETERS

    train_file = config.train_file
    test_file = config.test_file
    submit_file_dir = config.submit_file_dir
    submit_file_name = config.submit_file_name

    # index
    ID_column = config.ID_column

    # 予測対象
    target_value = config.target_value
    
    # カテゴリ変数をリストで設定
    ohe_columns = config.ohe_columns

    # 除外リスト
    score_reject_items = config.score_reject_items
    train_reject_items = config.train_reject_items

    pipelines = pipeline_config.pipelines
    gs_params = pipeline_config.gs_params


    # MAIN PROC
    main()