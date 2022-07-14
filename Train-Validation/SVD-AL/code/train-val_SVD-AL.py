def pred_generic(parameters):
  return parameters[0].loc[parameters[1]][parameters[1]]
  
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
from sklearn.decomposition import TruncatedSVD as SVD

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

new_df_0 = df_user_item.fillna(0)

new_df_globalAVG = df_user_item.fillna(global_avg)

new_df_ItemsAVG = df_user_item.fillna(value = items_avg.to_dict())

new_df_UserAVG = df_user_item.T.fillna(value = user_avg.to_dict())
new_df_UserAVG = new_df_UserAVG.T

dic_dfs = {
    'inputGlobalMean' : new_df_globalAVG,
    'inputItemMean' : new_df_ItemsAVG,
    'inputUserMean' : new_df_UserAVG,
    'input0' : new_df_0
}

for input_type in dic_dfs.keys():
	new_df = dic_dfs[input_type]
	print('Input type= ' + str(input_type))
	for factors in [2, 5, 10, 50, 100, 500, 1000, 2000]:
		
		start_train = time.time()
		
		print('factors = ' + str(factors))
		
		svd = SVD(n_components=factors, random_state=42)
		
		data_reducted = svd.fit_transform(new_df)
		
		end_train = time.time()
		
		time_train = end_train - start_train
		
		start_validation  = time.time()
		
		data_original_svd = svd.inverse_transform(data_reducted)

		df_original_svd = pd.DataFrame(data_original_svd, index=df_user_item.index, columns=df_user_item.columns)
		
		y_pred = generate([df_original_svd])

		ratings = df_validation.rating.to_list()

		rmse = MSE(ratings, y_pred, squared=False)

		end_validation = time.time()
		
		time_validation = end_validation - start_validation
		
		print("Input type=" + str(input_type) + " | factors: " + str(factors) + " | rmse: "+ str(rmse) + " | time_train: " + str(time_train) + " | time_validation: " + str(time_validation))
		dic_results[index] = [input_type,factors,rmse,time_train,time_validation] 

		index+=1

df_results = pd.DataFrame.from_dict(dic_results,  orient='index', columns = ['input type', 'factors', 'RMSE', 'Time Train', 'Time Validation'])
df_results.to_csv('../results/results_SVD-AL.csv', sep=';')
	
