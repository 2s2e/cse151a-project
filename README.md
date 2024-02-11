# Milestone 2 README

## Abstract:

The purpose of our machine learning application is to allow the user to create very specific playlists that suit their needs. The dataset we’ve selected contains a rich set of features such as, but not limited to the song album, artist, tempo, and instrumentalness. We aim to build a recommender system that first learns the user’s preferences and song associations and then allows them to provide a very specific description of a playlist they’d like for the model to generate. For example, a user could ask for a playlist that contains a large amount of lyrics, starts slow, but has faster songs as the playlist progresses. We foresee the use of both unsupervised learning, for grouping similar songs, and supervised learning, for adjusting to the user’s preferences.

## Data: 

Our dataset consists of a list of 17841 unique songs that were pulled from Spotify and YouTube. It contains labeling data such as the album, the artist, the rank, the name of the track, etc., and data on the song itself like key, instrumentalness, danceability, energy, acoustics, etc. Each row contains 26 features.  It can be found [here](https://github.com/2s2e/cse151a-project/blob/main/Spotify_Youtube.csv).


## Preprocessing:
Irrelevant features such as Spotify/Youtube URLs, Album titles, and whether or not they are Licensed were dropped.  In regards to training and from a user standpoint, these are irrelevant to music recommendations.
Relevant categorical data features were encoded.  For example, we one-hot encoded the Key of the Song and Album Type (Single, Album, Compilation).  This way we can incorporate relevant categorical details for training purposes.
Certain data including views, comments, duration, likes, and # of streams whose quantities are large integers are heavily skewed. To address this, we first logarithmically scaled the data, which resulted in a normal distribution, and then normalized the data to be between 0 and 1. 
Strings, such as artist name and album name were encoded based on the total number of views associated with that artist or album. Since using a number to represent a string would imply some kind of ordering, we figured we may as well use some kind of meaningful value to order the artists/albums. In this case, that value was the number of views. 

