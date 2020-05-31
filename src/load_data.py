# カテゴリ変数のカラムデータタイプを固定する
import pandas as pd

def dtype_columns(ohe_columns):
    return {k: object for k in ohe_columns}
    

# データの呼び出し
def data_load(train_file, reject_cols, my_dtype, target_value):
    # csvファイルからの読み出し
    dataset = pd.read_csv(train_file,
                        header=0,    # CSVデータの1行目が見出し(header)で有ることを指定。データは1行目が[0]
                        dtype=my_dtype)

    # 1列目のID情報、推論対象は特徴量から削除
    X = pd.DataFrame(dataset).drop(columns=reject_cols, axis=1)

    if target_value in dataset.columns.values:
        y = pd.Series(dataset[target_value])

    else:
        y = None

    return dataset, X, y