import json
import pandas as pd
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import re
import collections
import os
import datetime
from sklearn.cross_decomposition import CCA
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from scipy.stats import pearsonr
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor

def cca_similarity(X_cca,Y_cca):
    cca = CCA(n_components=1)
    cca.fit(X_cca, Y_cca)

    X_c, Y_c = cca.transform(X_cca, Y_cca)
    return (pearsonr(X_c.flatten(),Y_c.flatten())[0])
    
def pca_process (X,n_com):
    scaler = StandardScaler()
    X_0 = scaler.fit_transform(X) 
    pca = decomposition.PCA(n_components=n_com)
    pca.fit(X_0)
    X_0 = pca.transform(X_0)
    return X_0
def pca_similarity (X,Y):
    return pearsonr(pca_process(X).flatten(),pca_process(Y).flatten())[0]

def matrix_norm (X,Y,n_com):
    return np.linalg.norm(pca_process(X,n_com).flatten() - pca_process(Y,n_com).flatten())
def average_norm (X,Y):
    return np.linalg.norm(np.array(X.T.mean()) - np.array(Y.T.mean()))
    
def similarity_vector(df, pid, method):
    X_cca = df.drop(columns=['playlist_pid','track_uri']).T
    similarity_list = []
    list_temp = unique_playlist.copy()
    list_temp.remove(pid)
#     print (unique_playlist)
    for j in list_temp:
        Y_cca = df_similarity[df_similarity['playlist_pid'] == j]
        Y_cca = Y_cca.drop(columns=['playlist_pid','track_uri']).T
        try:
            if method == 'pca':
                sim = pca_similarity(X_cca,Y_cca)
            elif method == 'matrix norms':
                sim = matrix_norm(X_cca,Y_cca,2)
            elif method == 'cca':
                sim = cca_similarity(X_cca,Y_cca)
            elif method == 'average':
                sim = average_norm(X_cca,Y_cca)
                
            similarity_list.append({'pid_1':pid, 'pid_2':j, 'similarity': sim})
                
        except Exception as e:
            print (e)
            print (pid,j)
    return similarity_list
    
    
def get_pca_comp (df,n_com):
    X_cca = df
    X_cca = X_cca.drop(columns=['playlist_pid','track_uri']).T
    return pca_process(X_cca,n_com)
    
    
def preprocess (df):
    df_temp = df.copy()
    scaler = StandardScaler()
    df_temp = df_temp.drop(columns=['playlist_pid','track_uri'])
    data_tranformed = df_temp.T
    data_tranformed = scaler.fit_transform(data_tranformed) 
    df_temp = pd.DataFrame(data_tranformed.T, index=df_temp.index,columns=df_temp.columns)
    df_temp['playlist_pid'] = df.playlist_pid
    df_temp['track_uri'] = df.track_uri
    return df_temp
    
def track_similarity (df, lst_playlist):
    playlist_feature = get_pca_comp (df,1).flatten().T
#     print (playlist_feature)
    track_sim = []
    for i in lst_playlist:
        df_temp = df_similarity[df_similarity['playlist_pid'] == i]
        df_temp = preprocess(df_temp)
        for j,r in df_temp.iterrows():
            r_temp = r.copy()
            r_temp = r_temp.drop(labels=['playlist_pid','track_uri'])
            track_sim.append({'track_uri':r['track_uri'], "similarity": np.linalg.norm(playlist_feature - r_temp)})
    return track_sim

def r_precision (G,R):
    set1 = set(G)
    set2 = set(R)
    set_inter = set2.intersection(set1)
    return len(set_inter)/len(G), list(set_inter)

def cv_r_precision (pid):
    score_cv = []
    for j in range(5):
        df_groundtruth = df_similarity[df_similarity['playlist_pid'] == pid]
        X, y = train_test_split(df_groundtruth, test_size=0.3, random_state=j)
        l = similarity_vector(X,pid, 'matrix norms')
        df_sim_result = pd.DataFrame(l)
        top_sim_list = list(df_sim_result.sort_values("similarity").head(100).pid_2)+list(df_sim_result.sort_values("similarity").tail(10).pid_2)
        track_similar = track_similarity(X,top_sim_list)
        df_track = pd.DataFrame(track_similar)
        lst1 = list(y.track_uri) 
        lst2 = list(df_track.sort_values("similarity").head(400).track_uri) + list(df_track.sort_values("similarity").tail(100).track_uri)
        score, _ = r_precision(lst1,lst2)
        score_cv.append(score)
        print ('done with',j,  'iteration for playlist:', pid)
    return [np.mean(score_cv),pid]
    
    
df = pd.read_csv('data_withmeta/combined_data_withmeta.csv')
feature_list = ['danceability', 'energy', 'key', 'loudness', 'mode', \
       'speechiness', 'acousticness', 'instrumentalness', 'liveness', \
       'valence', 'tempo']
r_scores = []
unique_playlist = list(df['playlist_pid'].unique())
df_similarity = df[feature_list + ['playlist_pid','track_uri']]
df_similarity = df_similarity.dropna()

r_scores = []
f = open("result.txt", "w")
with ProcessPoolExecutor(max_workers=6) as executor:
    for r in executor.map(cv_r_precision, unique_playlist):
        r_scores.append(r[0])
        s = str(r[1]) + ',' + str(r[0]) + '\n'
        print (s)
        f.write(s)
avg_rp = np.mean(r_scores)
f.close()
f = open("final_score.txt", "w")
f.write(str(avg_rp))
f.close()
print (avg_rp)

