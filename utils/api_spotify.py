import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from urllib.request import urlopen
from sklearn import preprocessing
from sklearn.cluster import KMeans, Birch, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth

## Functions ##

# API Calls #
print('model initated')
#ADD LOCAL ENVIRONMENT ######################################

redirect_uri = 'http://127.0.0.1:9090'
username = 'michaelmarzec11'
scope = 'user-library-read playlist-modify-public'
#############################################################

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=spotify_cid, client_secret=spotify_secret, redirect_uri=redirect_uri, scope=scope))

liked_songs_df = pd.DataFrame()
liked_songs_features_df = pd.DataFrame()

api_limit = 50
total_songs = sp.current_user_saved_tracks(limit=api_limit, offset=0)['total']

offset = 0
while offset < total_songs: # 99
	liked_songs = sp.current_user_saved_tracks(limit=api_limit, offset=offset)
	df_add = pd.json_normalize(liked_songs['items']).reset_index()
	liked_songs_df = pd.concat([liked_songs_df, df_add])
	
	track_list = df_add['track.uri'].tolist()
	track_features = sp.audio_features(tracks=track_list) # https://developer.spotify.com/documentation/web-api/reference/#/operations/get-several-audio-features # audio_analysis --> save for later / second try?
	track_features = pd.json_normalize(track_features)
	liked_songs_features_df = pd.concat([liked_songs_features_df, track_features])

	offset += api_limit
	
liked_songs_features_df = liked_songs_features_df.reset_index(drop=True)

## append track name + artist for personal convenience <--- make optional function (takes a while... is there a more efficient way to do this via the api?)
# track_names = []
# track_artists = []

# for track_id in liked_songs_features_df['id'].tolist():
# 	track_name = sp.track(track_id)["name"]
# 	track_artist = sp.track(track_id)["artists"][0]['name']

# 	track_names.append(track_name)
# 	track_artists.append(track_artist)

# liked_songs_features_df['track_name'] = track_names
# liked_songs_features_df['arist_name'] = track_artists

# liked_songs_features_df.to_csv('features.csv') # probably convert to parquet once i want to remove duplicative efforts

# Prep for PCA / Clustering (reduce, standardize)
columns_for_clustering = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','time_signature'] # duraion_ms???
v2_columns_for_clustering = ['danceability','energy','loudness','acousticness','instrumentalness','liveness','valence','tempo'] # duraion_ms???
scaled_df = liked_songs_features_df[v2_columns_for_clustering]
scaler = preprocessing.StandardScaler().fit(scaled_df)
scaled_df = scaler.transform(scaled_df)

# pca = PCA(2)
# pca = PCA(n_components=9)
pca = PCA(0.95)
PCA_components = pca.fit_transform(scaled_df)
PCA_components = pd.DataFrame(PCA_components)

# # PCA Scree Plot --> Currently, implies 9 componenets due to marginal improvement at 10 and reaches ~95% variance explained
# principalDf = pd.DataFrame(data = PCA_components)
# PC_values = np.arange(pca.n_components_) + 1
# plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Proportion of Variance Explained')
# plt.show()

# # visualize 2-dPCA
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# ax.scatter(principalDf.iloc[:,0], principalDf.iloc[:,1])
# ax.grid()
# plt.show()


## Cluster Analysis (inertia plot: elbow analysis)
# inertias = []
# for k in range(1,26):
#     model = KMeans(n_clusters=k)
    
#     # Fit model to samples
#     model.fit(PCA_components)
    
#     # Append the inertia to the list of inertias
#     inertias.append(model.inertia_)
    
# plt.plot(range(1,26), inertias, '-p', color='gold')
# plt.xlabel('number of clusters, k')
# plt.ylabel('inertia')
# plt.show()
# breakpoint()


# Clustering: https://machinelearningmastery.com/clustering-algorithms-with-python/
# https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
# K-Means
# 11 Clusters based on scree
model = KMeans(n_clusters=20)
model.fit(PCA_components)
labels = model.predict(PCA_components)

# ## Birch
# ## 11 Clusters based on scree
# model = Birch(threshold=0.01, n_clusters=11)
# model.fit(PCA_components)
# labels = model.predict(PCA_components)

# ## DBSCAN
# ## 11 Clusters based on scree
# model = DBSCAN(eps=0.3, min_samples=9) # min samples allows for minimum # in cluster (songs in playlist ... chil) ... eps, maybe we try gridsearchCV?
# # model.fit(PCA_components)
# labels = model.fit_predict(PCA_components)

# # Gaussian
# # 11 Clusters based on scree
# model = GaussianMixture(n_components=11)
# model.fit(PCA_components)
# labels = model.fit_predict(PCA_components)


plt.scatter(PCA_components[0], PCA_components[1], c=labels)
plt.savefig('v8_fig.png')
# # plt.show()
# breakpoint()



PCA_components['labels'] = labels
# PCA_components.to_csv('labels.csv')

song_labels = pd.DataFrame(liked_songs_features_df) 
song_labels['labels'] = labels
song_labels.to_csv('v8_song_labels.csv')
breakpoint()

### delete playlists




### create playlists
total_playlists = song_labels['labels'].max()

for x in range(total_playlists + 1):
	playlist_name = "V8_autoPlaylist_" + str(x)
	playlist_result = sp.user_playlist_create(username, name=playlist_name)
	playlist_id = playlist_result['external_urls']['spotify']

	track_ids = song_labels[song_labels['labels']==x]['id'].tolist()

	add_max = 100
	current_track_low = 0
	current_track_high = 100
	if len(track_ids) > add_max:
		while current_track_low < len(track_ids):
			current_ids = track_ids[current_track_low:current_track_high]
			sp.user_playlist_add_tracks(username, playlist_id, current_ids)
			current_track_low += add_max
			current_track_high += add_max
	else:
		sp.user_playlist_add_tracks(username, playlist_id, track_ids)




