def SVD_GD(df, user_len, items_len ,factors, global_avg, iters, learning_rate, reg):

  b_u = pd.Series(np.zeros(user_len), index=df.user_id.unique())
  b_i = pd.Series(np.zeros(items_len), index=df.movie_id.unique())

  P = pd.DataFrame(np.zeros((user_len, factors)), index = df.user_id.unique())
  Q = pd.DataFrame(np.zeros((items_len, factors)), index = df.movie_id.unique())

  for iter in range(iters):
    for item,row in df.iterrows(): 
      user = row['user_id']
      item = row['movie_id']
      rating = row['rating']

      pred = global_avg + b_u.loc[user] + b_i.loc[item] + np.dot(P.loc[user],Q.loc[item])

      e_ui = rating - pred
      b_u.loc[user] = b_u.loc[user] + (learning_rate*e_ui)
      b_i.loc[item] = b_i.loc[item] + (learning_rate*e_ui)
      for factor in range(factors):
        temp_puf = P.loc[user][factor]
        P.loc[user][factor] = P.loc[user][factor] + (learning_rate * ((e_ui*Q.loc[item][factor]) - (reg*P.loc[user][factor])))
        Q.loc[item][factor] = Q.loc[item][factor] + (learning_rate * ((e_ui*temp_puf) - (reg*Q.loc[item][factor])))
        
  return b_u, b_i, P, Q     
  
def pred_generic(global_avg, b_u, b_i, P, Q, user, item):
  return global_avg + b_u[user] + b_i[item] + np.dot(P.loc[user],Q.loc[item])
  
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

      value = pred_generic(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4], parameters[5], parameters[6])
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

dic_results = {}
index = 0

df_train = pd.read_csv('../dataset/train_data .csv')

df_train_data, df_validation = train_test_split(df_train, test_size=0.01, random_state=42, shuffle=True)

df_user_item = pd.DataFrame(index = df_train_data['user_id'].unique(), columns=df_train_data['movie_id'].unique())

for movie in df_train_data['movie_id'].unique():
  temp = df_train_data[df_train_data.movie_id == movie]
  df_user_item.loc[temp.user_id, movie] = temp.rating.to_list()
  
items_avg = np.mean(df_user_item)
user_avg = np.mean(df_user_item.T)
global_avg = np.nanmean(df_user_item.values.tolist())

user_len = len(df_train_data.user_id.unique())
items_len = len(df_train_data.movie_id.unique())

reg = 0.01

for factors in [2, 5, 10]:
	print('factors: ' + str(factors))
	for iters in [10, 20]:
		print('iters: ' + str(iters))
		for lr in [0.001, 0.01, 0.1]:
			print('lr: ' + str(lr))

			start_train = time.time()

			b_u,b_i,P,Q = SVD_GD(df_train_data, user_len, items_len, factors, global_avg, iters, lr, reg)

			end_train = time.time()
				
			time_train = end_train - start_train

			start_validation  = time.time()
				
			y_pred = generate([global_avg, b_u, b_i, P, Q])

			ratings = df_validation.rating.to_list()

			rmse = MSE(ratings, y_pred, squared=False)

			end_validation = time.time()

			time_validation = end_validation - start_validation
				
			print('factors: ' + str(factors) + 'iters: ' + str(iters) + 'lr: ' + str(lr) + " | rmse: "+ str(rmse) + " | time_train: " + str(time_train) + "time_validation: " + str(time_validation))

			dic_results[index] = [factors,iters,lr,rmse,time_train,time_validation] 

			index+=1

df_results = pd.DataFrame.from_dict(dic_results,  orient='index', columns = ['factors', 'iters','lr', 'Time Train', 'Time Validation'])
df_results.to_csv('../results/results_SVD-GD.csv', sep=';')
	
    	
      
