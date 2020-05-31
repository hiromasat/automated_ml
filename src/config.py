import time

# SET PARAMETERS
train_file = '../data/final_hr_analysis_train.csv'
test_file = '../data/final_hr_analysis_test.csv'
submit_file_dir = '../submit/'
submit_file_name = 'submit_tabuchi'+str(time.time()) +'.csv'

# index
ID_column = 'index'

# 予測対象
target_value = 'left'
    
# カテゴリ変数をリストで設定
ohe_columns = ['sales','salary,']

# 除外リスト
score_reject_items =[ID_column]
train_reject_items = score_reject_items + [target_value]

