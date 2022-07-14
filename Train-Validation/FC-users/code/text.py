import pickle
with open('../parcial_similarities/dic_sim_users_pearson.p', 'rb') as fp:
	data = pickle.load(fp)

print(data.keys())
