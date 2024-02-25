# Milestone 2 README

## Abstract:

The purpose of our machine learning application is to allow the user to create very specific playlists that suit their needs. The dataset we’ve selected contains a rich set of features such as, but not limited to the song album, artist, tempo, and instrumentalness. We aim to build a recommender system that first learns the user’s preferences and song associations and then allows them to provide a very specific description of a playlist they’d like for the model to generate. For example, a user could ask for a playlist that contains a large amount of lyrics, starts slow, but has faster songs as the playlist progresses. We foresee the use of both unsupervised learning, for grouping similar songs, and supervised learning, for adjusting to the user’s preferences.

## Data: 

Our dataset consists of a list of 17841 unique songs that were pulled from Spotify and YouTube. It contains labeling data such as the album, the artist, the rank, the name of the track, etc., and data on the song itself like key, instrumentalness, danceability, energy, acoustics, etc. Each row contains 26 features.  It can be found [here](https://github.com/2s2e/cse151a-project/blob/main/Spotify_Youtube.csv).


## Preprocessing:
Irrelevant features such as Spotify/Youtube URLs, Album titles, and whether or not they are Licensed were dropped.  In regards to training and from a user standpoint, these are irrelevant to music recommendations.
Relevant categorical data features were encoded.  For example, we one-hot encoded the Key of the Song and Album Type (Single, Album, Compilation).  This way we can incorporate relevant categorical details for training purposes.
Certain data including views, comments, duration, likes, and # of streams whose quantities are large integers are heavily skewed. To address this, we first logarithmically scaled the data, which resulted in a normal distribution, and then normalized the data to be between 0 and 1. 
Strings, such as artist name and album name were encoded based on the total number of views associated with that artist or album. Since using a number to represent a string would imply some kind of ordering, we figured we may as well use some kind of meaningful value to order the artists/albums. In this case, that value was the number of views.  Our preprocessing can be found [here](https://github.com/2s2e/cse151a-project/blob/main/Milestone.ipynb)

## Train your first model

For our first model we tested out one of the most popular unsupervised learning techniques – K Means Clustering. To identify the number of clusters in the model we used the silhouette score metric. 
Our first model can be found under the clustering section (below data preprocessing) [here](https://github.com/2s2e/cse151a-project/blob/main/Milestone%203.ipynb).

The following is how K means clustering works:
Centroid for each of the clusters are uniformly initialized
We find the distance of each of the samples to all of the centroids
Based on the centroid a point is closest to, it is assigned to the corresponding cluster associated with that centroid
Based on the obtained clusters, the median of each component corresponding to points in a cluster is used to obtain the new centroids
The process is repeated until the process converges

After training the K Means clustering model, we used evaluation metrics like the WCSS to see how the model performs as well.
## Evaluate your model compare training vs test error

We find that our training error and testing error, as measured by WCSS, are very similar. This suggests that our model is neither overfitting nor underfitting, as either of these situations should suggest a larger discrepancy between our errors.

## Where does your model fit in the fitting graph.

As stated before, the training and test error for our model is very similar. This suggests that our model is neither overfitting nor underfitting, as either of these situations would suggest a larger discrepancy between our errors. Thus, on the fitting graph, we are in a well-situated spot.

## What are the next 2 models you are thinking of and why?

One model we are thinking of developing is a DBSCAN model.  Similar to K-Means, it will allow us to perform unsupervised clustering to group songs of similar qualities together.  However, the benefit of DBSCAN is that we will no longer have to specify how many clusters we expect to see.  Given the diverse dataset of music, it will potentially be beneficial to allow the algorithm to try to determine the number of clusters instead of us as developers trying to force a pattern of some explicit number of clusters.

Another model we are thinking of is the K-Nearest-Neighbor model. In contrast to K-Means, KNN finds the k nearest neighbors based on the distance metric. One benefit of KNN is that it’s less sensitive to outliers compared to K-mean clustering. Outliers are less likely to affect the clustering results significantly because KNN considers only the nearest neighbors when assigning points to clusters. 


## Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?

Note: Please make sure preprocessing is complete and your first model has been trained, and predictions for train, val and test are done and analyzed. 

We have applied a frequency-based encoding approach to both artists and albums, this, combined with KMeans clustering, is our team’s first attempt to come up with an innovative approach to build a recommendation system. This method leverages the popularity of artists and albums to align recommendations with mainstream user preferences.

However, here is a critical challenge we are facing right now:
We want to ensure that this recommendation system can put forward recommendations that span a diverse range of musical genres, and not overly concentrated with a few overly populous clusters.

Here are some improvements on the model we have found to be reasonable:

* Dynamic clustering algorithms like DBSCAN, this has the potential to recommend a more natural groupings of songs
* Develop a mechanism for users to give feedback to rate their recommendations, and this should allow the model to refine its predictions over time and be adaptive to the current musical taste of the user population
* Incorporate more features, and try encoding schemas beyond frequency encoding
