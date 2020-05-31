import pandas as pd

# データのステータスプリント
def print_on_terminal(title, dataset, X, y):
    # check the shape
    print(title)
    print('----------------------------------------------------------------')
    print('X shape: (%i,%i)' %X.shape)
    print('----------------------------------------------------------------')
    print('y shape: (%i,)' %y.shape)
    print('----------------------------------------------------------------')
    print(y.value_counts())
    print('left（1：退職、0：非退職の正解ラベル）')
    print('----------------------------------------------------------------')
    print()
    # 教師データとするｙの型
    print('y.shape', y.shape)

def make_submission_file(ID_column, dataset, target_value, y_pred, submit_file_dir, submit_file_name):
    # 提出用のファイルに書き出し
    my_result = pd.DataFrame({ID_column:dataset[ID_column], target_value:y_pred})
    my_result.to_csv(submit_file_dir+submit_file_name, index=False)
