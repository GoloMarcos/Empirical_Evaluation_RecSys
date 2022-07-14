def similarity_itens(similarities, neighbor_item, item):
  
  if (item,neighbor_item) not in similarities.keys():
    similarity = similarities[(neighbor_item,item)]
  else:
    similarity = similarities[(item,neighbor_item)]
    
  return similarity
  
def k_neighbors_nearest(df, similarities, k, user, item):
  k_neighbors_similarity = [-2] * k
  k_neighbors = [-1] * k

  for neighbor_item in df.columns:
    
    ni = neighbor_item
    if item != ni and df.loc[user][ni] is not np.nan:
      
      similarity = similarity_itens(similarities, ni, item)

      for i in range(k):
        if similarity > k_neighbors_similarity[i]:
          aux = k_neighbors_similarity[i]
          k_neighbors_similarity[i] = similarity
          similarity = aux

          aux = k_neighbors[i]
          k_neighbors[i] = ni
          ni = aux

  return k_neighbors

def pred(df, similarities, k, user, item):

  k_neighbors = k_neighbors_nearest(df, similarities, k, user, item)
  
  sum = 0
  sum_similarity = 0

  for neighbor_item in k_neighbors:
    if(neighbor_item != -1): #se nao deu o numero maximo de vizinhos mais proximos
      rating_neighbor_item = df.loc[user][neighbor_item]
    
      similarity = similarity_itens(similarities, neighbor_item, item)

      sum_similarity+= similarity
      
      sum+= similarity * rating_neighbor_item
  
  if sum_similarity ==0 and sum ==0:
    return 0
  elif sum_similarity ==0 and sum !=0:
    return sum
  else:
    return sum/sum_similarity

def pred_generic(parameters):
  return pred(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4])
  
def generate(parameters):
  l= []
  
  for item,row in tqdm(df_validation.iterrows()):
    user = row['user_id']
    item = row['movie_id']
    if user not in df_train_data['user_id'].unique() and item in df_train_data['movie_id'].unique():
      l.append(items_mean[item])
    elif user in df_train_data['user_id'].unique() and item not in df_train_data['movie_id'].unique():
      l.append(user_mean[user])
    elif user not in df_train_data['user_id'].unique() and item not in df_train_data['movie_id'].unique():
      l.append(global_mean)
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
  
import pickle
import time
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split
from tqdm import tqdm

with open('../parcial_similarities/dic_sim_items_genres_jaccard.p', 'rb') as fp:
	data = pickle.load(fp)

dic_results = {}
index = 0

df_train = pd.read_csv('../dataset/train_data .csv')

df_train_data, df_validation = train_test_split(df_train, test_size=0.01, random_state=42, shuffle=True)

df_user_item = pd.DataFrame(index = df_train_data['user_id'].unique(), columns=df_train_data['movie_id'].unique())

for movie in df_train_data['movie_id'].unique():
  temp = df_train_data[df_train_data.movie_id == movie]
  df_user_item.loc[temp.user_id, movie] = temp.rating.to_list()

items_mean = np.mean(df_user_item)
user_mean = np.mean(df_user_item.T)
global_mean = np.nanmean(df_user_item.values.tolist())

for k in range(1,50):
  start = time.time()
  print('k = ' + str(k))
  
  y_pred = generate([df_user_item, data, k])
  
  ratings = df_validation.rating.to_list()

  rmse = MSE(ratings, y_pred, squared=False)

  end = time.time()
  time_ = end - start
  print("k: " + str(k) + " | rmse: "+ str(rmse) + " | time: " + str(time_))
  dic_results[index] = [k,rmse,time_] 

  index+=1

df_results = pd.DataFrame.from_dict(dic_results,  orient='index', columns = ['k', 'RMSE', 'Time'])
df_results.to_csv('../results/results_FBCKNN_genres.csv', sep=';')
	
