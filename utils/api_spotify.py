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
#ADD To LOCAL ENVIRONMENT #####################################
redirect_uri = 'http://127.0.0.1:9090'
username = 'michaelmarzec11'
scope = 'user-library-read playlist-modify-public'
############################################################

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=spotify_cid, client_secret=spotify_secret, redirect_uri=redirect_uri, scope=scope))

def create_audio_featres_df(api_limit=50, offset=0):
	liked_songs_df = pd.DataFrame()
	liked_songs_features_df = pd.DataFrame()

	total_songs = sp.current_user_saved_tracks(limit=api_limit, offset=0)['total']

	while offset < total_songs:
		liked_songs = sp.current_user_saved_tracks(limit=api_limit, offset=offset)
		df_add = pd.json_normalize(liked_songs['items']).reset_index()
		liked_songs_df = pd.concat([liked_songs_df, df_add])
		
		track_list = df_add['track.uri'].tolist()
		track_features = sp.audio_features(tracks=track_list) # https://developer.spotify.com/documentation/web-api/reference/#/operations/get-several-audio-features # audio_analysis --> save for later / second try?
		track_features = pd.json_normalize(track_features)
		liked_songs_features_df = pd.concat([liked_songs_features_df, track_features])

		offset += api_limit
		
	liked_songs_features_df = liked_songs_features_df.reset_index(drop=True)
	return liked_songs_features_df

def append_trackName_artist(append_liked_songs_df):# append track name + artist for personal convenience <--- make optional function (takes a while... is there a more efficient way to do this via the api?)
	track_names = []
	track_artists = []

	for track_id in append_liked_songs_df['id'].tolist():
		track_name = sp.track(track_id)["name"]
		track_artist = sp.track(track_id)["artists"][0]['name']

		track_names.append(track_name)
		track_artists.append(track_artist)

	append_liked_songs_df['track_name'] = track_names
	append_liked_songs_df['arist_name'] = track_artists
	return append_liked_songs_df

def PCA_execute(df_pca_prep, threshold=0.90): # Prep for PCA / Clustering (reduce, standardize)
	columns_for_clustering = ['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','time_signature'] # duraion_ms???
	v2_columns_for_clustering = ['danceability','energy','loudness','acousticness','instrumentalness','liveness','valence','tempo']
	
	scaled_df = df_pca_prep[v2_columns_for_clustering]
	scaler = preprocessing.StandardScaler().fit(scaled_df)
	scaled_df = scaler.transform(scaled_df)

	# pca = PCA(2)
	# pca = PCA(n_components=9) # 9 componenets due to marginal improvement at 10 and reaches ~95% variance explained
	pca = PCA(threshold)
	pca_comps = pca.fit_transform(scaled_df)
	pca_comps = pd.DataFrame(pca_comps)
	return pca_comps, pca

def pca_scree_plot(pca_comps, pca, plt_save=False, png_name='screePlot.png', plt_show=False): #PCA Scree Plot --> Currently, implies 9
	principalDf = pd.DataFrame(data = pca_comps)
	PC_values = np.arange(pca.n_components_) + 1
	plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
	plt.title('Scree Plot: 0.999 Variance Threshold')
	plt.xlabel('Principal Component')
	plt.ylabel('Proportion of Variance Explained')
	if plt_save == True:
		plt.savefig(png_name)
	if plt_show == True:
		plt.show()


def intertia_plot(pca_comps, no_clusters=26, plt_save=False, png_name='inertiaPlot.png', plt_show=False):# Cluster Analysis (inertia plot: elbow analysis)
	inertias = []
	for k in range(1,no_clusters):
	    model = KMeans(n_clusters=k)
	    
	    # Fit model to samples
	    model.fit(pca_comps)
	    
	    # Append the inertia to the list of inertias
	    inertias.append(model.inertia_)
	    
	plt.plot(range(1,no_clusters), inertias, '-p', color='gold')
	plt.xlabel('number of clusters, k')
	plt.ylabel('inertia')
	if plt_save == True:
		plt.savefig(png_name)
	if plt_show == True:
		plt.show()

# Clustering: https://machinelearningmastery.com/clustering-algorithms-with-python/
# https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
def cluster_algo(pca_comps, liked_songs_df, cluster_num=20, kmeans=False, birch=False, dbscan=False, gaussian=False):
	# K-Means
	if kmeans==True:
		model = KMeans(n_clusters=cluster_num)
		model.fit(pca_comps)
		labels = model.predict(pca_comps)

	# Birch
	if birch==True:
		model = Birch(threshold=0.01, n_clusters=cluster_num)
		model.fit(pca_comps)
		labels = model.predict(pca_comps)

	# DBSCAN
	if dbscan==True:
		model = DBSCAN(eps=0.3, min_samples=9) # min samples allows for minimum # in cluster (songs in playlist ... chil) ... eps, maybe we try gridsearchCV?
		model.fit(pca_comps)
		labels = model.fit_predict(pca_comps)

	# Gaussian
	if gaussian==True:
		model = GaussianMixture(n_components=cluster_num)
		model.fit(pca_comps)
		labels = model.fit_predict(pca_comps)
	
	pca_comps['labels'] = labels

	# add labels to original dataset
	song_labels = pd.DataFrame(liked_songs_df) 
	song_labels['labels'] = labels
	# song_labels.to_csv('v8_song_labels.csv')
	return pca_comps, song_labels, labels

def cluster_scatter_plot(pca_comps, labels, plt_save=False, png_name='cluster_scatter_plot.png', plt_show=False):
	plt.scatter(pca_comps[0], pca_comps[1], c=labels)
	plt.title('V6: K-Means Clustering Results')
	plt.savefig(png_name)
	if plt_save == True:
		plt.savefig(png_name)
	if plt_show == True:
		plt.show()

def create_playlist(song_labels, username, add_max=100, current_track_low=0, current_track_high=100):
	total_playlists = song_labels['labels'].max()

	for x in range(total_playlists + 1):
		playlist_name = "autoPlaylist_" + str(x)
		playlist_result = sp.user_playlist_create(username, name=playlist_name)
		playlist_id = playlist_result['external_urls']['spotify']

		track_ids = song_labels[song_labels['labels']==x]['id'].tolist()
		
		if len(track_ids) > add_max:
			while current_track_low < len(track_ids):
				current_ids = track_ids[current_track_low:current_track_high]
				sp.user_playlist_add_tracks(username, playlist_id, current_ids)
				current_track_low += add_max
				current_track_high += add_max
		else:
			sp.user_playlist_add_tracks(username, playlist_id, track_ids)

def main():
	liked_songs_features_df = create_audio_featres_df()
	PCA_components, pca = PCA_execute(liked_songs_features_df, threshold=0.90) 
	# pca_scree_plot(PCA_components, pca, plt_show=False, plt_save=True, png_name='vX_screeplot.png')
	# intertia_plot(PCA_components, no_clusters=16, plt_show=False, plt_save=True, png_name='vX_inertiaplot.png')
	pca_clusters, song_labels_df, labels = cluster_algo(PCA_components, liked_songs_features_df, cluster_num=20, kmeans=True, birch=False, dbscan=False, gaussian=False)# should only choose one
	cluster_scatter_plot(PCA_components, labels, plt_save=True, png_name='vX_cluster_scatter_plot.png', plt_show=True)
	# create_playlist(song_labels_df, username)


if __name__ == '__main__':
	main()
	

