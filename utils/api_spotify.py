import datetime
import math
import numpy as np
import pandas as pd
import re
from urllib.request import urlopen
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth



## Functions ##

# API Calls #

cid = 'ce9c37183c77483f8d365f7731f0345f'
secret = '3c97cdd90ccf424c8d8e0fcddf9590a7'
redirect_uri = 'http://127.0.0.1:9090'

# scope = 'playlist-modify-public'
# sp_create_playlist = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid, client_secret=secret, redirect_uri=redirect_uri, scope=scope))
# sp_create_playlist.user_playlist_create('michaelmarzec11', name='test')

scope = 'user-library-read'
sp_read_liked_songs = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid, client_secret=secret, redirect_uri=redirect_uri, scope=scope))
sp_audio_features = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=cid, client_secret=secret, redirect_uri=redirect_uri))


liked_songs_df = pd.DataFrame()
liked_songs_features_df = pd.DataFrame()


api_limit = 50
total_songs = sp_read_liked_songs.current_user_saved_tracks(limit=api_limit, offset=0)['total']

offset = 0
while offset < total_songs:
# while offset <= 99:
	liked_songs = sp_read_liked_songs.current_user_saved_tracks(limit=api_limit, offset=offset)
	df_add = pd.json_normalize(liked_songs['items']).reset_index()
	liked_songs_df = pd.concat([liked_songs_df, df_add])
	
	track_list = df_add['track.uri'].tolist()
	track_features = sp_audio_features.audio_features(tracks=track_list) # https://developer.spotify.com/documentation/web-api/reference/#/operations/get-several-audio-features # audio_analysis --> save for later / second try?
	track_features = pd.json_normalize(track_features)
	liked_songs_features_df = pd.concat([liked_songs_features_df, track_features])

	offset += api_limit
	



