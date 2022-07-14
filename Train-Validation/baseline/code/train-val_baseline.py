def pred(df, const, user, item):

  bias_item = 0
  if item in df.columns:
    item_len = len(df[item][~df[item].isnull()])
    for rating_user_item in df[item][~df[item].isnull()]:
      bias_item = bias_item + (rating_user_item - global_avg)
  else:
    item_len = 0
  
  bias_item = bias_item/(const + item_len)

  bias_user = 0
  if user in df.index:
    user_len = len(df.loc[user][~df.loc[user].isnull()])
    for rating_user_item in df.loc[user][~df.loc[user].isnull()]:
      bias_user = bias_user + (rating_user_item - global_avg - bias_item)
  else:
    user_len = 0

  bias_user = bias_user/(const + user_len)

  rating = global_avg + bias_item + bias_user
  
  return rating
  
def pred_generic(parameters):
  return pred(parameters[0], parameters[1], parameters[2], parameters[3])
  
def generate(parameters):
  l= []
  
  for item,row in tqdm(df_validation.iterrows()):
    user = row['user_id']
    item = row['movie_id']
    if user not in df_train_data['user_id'].unique() and item in df_train_data['movie_id'].unique():
      l.append(items_avg[item])
    elif user in df_train_data['user_id'].unique() and item not in df_train_data['movie_id'].unique():
      l.append(user_avg[user])
    elif user not in df_train_data['user_id'].unique() and item not in df_train_data['movie_id'].unique():
      l.append(global_avg)
    else:
      parameters.append(user)
      parameters.append(item)

      value = pred_generic(parameters)
      if value < 1:
        value = 1
      if value > 5:
        value = 5

      l.append(value)
  
  return l

import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from tqdm import tqdm

df_train = pd.read_csv('../dataset/train_data .csv')

df_train_data, df_validation = train_test_split(df_train, test_size=0.01, random_state=42, shuffle=True)

df_user_item = pd.DataFrame(index = df_train_data['user_id'].unique(), columns=df_train_data['movie_id'].unique())

for movie in df_train_data['movie_id'].unique():
  temp = df_train_data[df_train_data.movie_id == movie]
  df_user_item.loc[temp.user_id, movie] = temp.rating.to_list()
  
items_avg = np.mean(df_user_item)
user_avg = np.mean(df_user_item.T)
global_avg = np.nanmean(df_user_item.values.tolist())

dic_results = {}
index = 0

for const in [0.00000000001, 0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
  start = time.time()
  print('const = ' + str(const))
  
  y_pred = generate([df_user_item, const])
  
  ratings = df_validation.rating.to_list()

  rmse = MSE(ratings, y_pred, squared=False)

  end = time.time()
  time_ = end - start
  print("lambda: " + str(const) + " | rmse: "+ str(rmse) + " | time: " + str(time_))
  dic_results[index] = [const,rmse,time_] 

  index+=1

df_results = pd.DataFrame.from_dict(dic_results,  orient='index', columns = ['lambda', 'RMSE', 'Time'])
df_results.to_csv('../results/results_baseline.csv', sep=';')
