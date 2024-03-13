# Results

To assess the quality of our clusters we used the WCSS (Within-Cluster Sum of Square) evalutation metric

### Model 1 Results

![image](https://github.com/2s2e/cse151a-project/assets/97645823/49cef2a6-0e13-4c20-9e16-ff9e14bc9264)

**Model 1 Fig 2. Radar Chart of Feature Across Clusters**
The radar chart provides a comparative visualization of the average values of five features—'Danceability', 'Energy', 'Loudness', 'Speechiness', and 'Acousticness'—across five different clusters identified within the dataset.
'Danceability' and 'Energy' axes show that clusters tend to have moderate to high average values, suggesting these are prominent characteristics across clusters. Remarkably ‘Danceability’ is lowest for cluster 4 and relatively low for cluster 2.
The 'Loudness' axis displays less variability between clusters, with most clusters exhibiting mid-range average loudness, with the exception of cluster 4, which has the lowest level of 'Loudness'.
'Speechiness' has lower average values across all clusters, indicating that tracks with a lot of spoken words are less common.

- Using the silhouette_score metric determined ideal number of clusters for K Means as `n_clusters=20`
- The WCSS scores were similar indicating the model did not overfit.
    - WCSS score on train: 0.026
    - WCSS Score on test: 0.027 
- On average, 3 out of the 5 attributes were in line with the user preferences
- Model was in a good spot on fitting graph.

### Model 2 Results

![image](https://github.com/2s2e/cse151a-project/assets/97645823/834d8180-4ece-4803-82df-92cfcf8d3591)
**Model 2 Fig 1. Histogram on Feature Distributions within Cluster 1**
'Danceability' in cluster 1 has a prominent peak, indicating a strong consistency in danceability among the tracks in this cluster, generally high.
The 'Energy' feature shows a wide distribution, suggesting a diverse range of energy levels within the tracks of cluster 1.
'Loudness' appears to be normally distributed with a slight right skew, indicating varied loudness but with a tendency towards higher loudness levels.
'Speechiness' displays a high frequency of lower values with a long tail to the right, similar to cluster 0, suggesting that vocal content is generally low in these tracks.
The histogram for 'Acousticness' displays a sharp peak at the lower end, which rapidly decreases, indicating that tracks in cluster 1 have a low degree of acousticness.


- The ideal value for the epsilon parameter in our DBSCANS model, as determined by the Elbow Score method, was found approximately towards the 0.05 to 0.1 range.
- An iterative search of hyperparameters min_samples and episilon yielded the ideal hyperparameters:
    -  `epsilon = 0.05` 
    -  `min_samples = 6`.
- The silhouette score yielded from said parameters was `0.56226`.
-  These parameters yield 161 unique clusters.
-  The WCSS scores were variagated between training and testing.
    -  WCSS Score on Train: 0.01
    -  WCSS Score on Test: 0.09
-  Model was significantly overfit.

### Model 3 Results

![image](https://github.com/2s2e/cse151a-project/assets/97645823/bedaca93-d2ff-48f3-985b-ddb4ebe76d66)

**Model 3 Fig 1. WCSS of Training Data for Values of k in KNN**
Blue Line: The line represents changes in the average WCSS of the training data as the number of neighbors (k) in the KNN algorithm increases.
Dots: Each dot corresponds to the average WCSS for a specific value of k, where k ranges from 1 to 91 in increments of 10.

![image](https://github.com/2s2e/cse151a-project/assets/97645823/e7206a2c-d7e5-475f-8e17-634e3bade1b5)

**Model 3 Fig 2.** WCSS of Testing Data for Values of k in KNN
Same as Model 3 Fig 1.

- `n_clusters=20` since we are using the same K means model
- K nearnest neighbors was implemented to have a custom `knn_predict` instead of using the in built `predict()` methods provided with K Means
- The WCSS scores were similar indicating the model did not overfit.
    - WCSS score on train: 0.0275
    - WCSS Score on test: 0.0279
- Similar performance relative to model 1

# Discussion
For our data, we only have information about the music tracks that were pulled from Spotify and YouTube, but no data in regards to genre nor type of music.  Thus, the first challenge we needed to overcome was to determine what statistical models we wanted to use and how to harness our data to make such decisions.

As such, we decided to start with unsupervised learning algorithms.  By using clustering algorithms to determine the relationship between observations, we realized we could use such groupings to generate labels, and as such provide a framework for predictions and further supervised learning. 

To begin this process, we decided to start with the clustering algorithm K-Means.  Our first step was to find the clusters.
While performing analysis using the silhouette_score, we wanted to determine the ideal number of clusters. The following should be kept in mind while doing so: 

1. Silhouette analysis can be used to study the separation distance between the resulting clusters. As we increase the number of clusters, the clusters tend to be located closer to one another it is expected for the silhouette_score to show a decreasing trend.
2. We have a bad pick if we have clusters with below average silhouette scores.
3. We do not one of the clusters to have significantly more samples than the other clusters.


Based on the analysis of silhouette_score we decided to pick the number of clusters as 20. After training the model, we decided to test it on user input.

The user would input values for 5 features. Some examples of these include danceability, acoustics, etc. Then, we would run that through our clustering algorithm to predict which cluster that song would belong to. After determining the cluster, we would give the user some song suggestions.
It was great to see that for most of the features, the user input was aligned with our predictions. However, they were not quite aligned for the same. This could have been a result of several reasons including the following:

1. The inherent complexity of building a recommendation system and an unsupervised learning algorithm
2. Some of the features may have been correlated. So, if the user input on those would present conflicting requirements on a set of positively correlated features, the model would have been biased for such correlated features.

Wrapping up our first model, as we were performing a cluster analysis, we aim to use the numerical data of Danceability, Energy, Loudness, Speechiness, and Acousticness (which will once again be accepted as user parameters) to predict a recommendation, trained on a DBSCAN model based on these parameters. As such, we used the same data and, like for the last clustering algorithm, given the lack of labels to verify, we ended up using metrics dedicated for clustering algorithms. 

Important metrics to choose a clustering model was silhouette score, and Within Cluster Sum of Squares to evaluate model performance on training and test data.  Whereas our implementation with K-Means in our first model required us to suggest the number of clusters, we tried DBSCANS, which evaluates the structure of the clustering, to determine an optimal number of clusters.

Naturally, the first step we decided to do was to tune our hyperparameters and evaluate on the metric silhouette score as a means to determine which hyperparameters may yield the best results.  While some modifications such as cross-validation and feature expansion were not relevant to our model for this milestone, we performed hyper-parameter tuning on our DBSCANS model. This was done by iteratively creating models based on epsilon values between some range, and min_samples between some ranges, effectively a simple GridSearch. We track the parameters that produce the clusters with the greatest silhouette score as our metric, suggesting the best fit clusters. Because we iteratively searched through different hyperparameters, we could track the history and effect of different parameter configurations on the cluster strength of our model (which wes evaluated via the metric of silhouette score). As such we were able to choose a more optimized clustering model, which had a score of approximately 0.56, which is a considerable performance increase than other hyperparameter configurations that could go as low as 0.30.

After optimizing our hyperparameters, we made predictions on our model and tested the WCSS scores of both our training and testing set.  We did this to check for potential underfitting/overfitting.  Our results showed that our model was significantly overfitting on the training data.  Thinking about why this might be the case, we realized that, in comparison to our K-Means model which was performed with 20 clusters, we find that the DBSCANS model is seemingly optimized at over 150 clusters. This is likely contributing to the loss of generality of our model, as clusters are far more detailed for the training data. One factor for this may be the choice of epsilon as a sort of "tolerance factor," which was derived by looking at the elbow graph based on the training data. Meanwhile, the first model was likely more tolerant to clusters (having only 20 to begin with), and as such was more generalizable. As such, ideas on how to improve our model would be a finer tuning on the epsilon hyperparameter to start, such as testing a larger range and with finer steps.  Indeed, given the large variety of music and the fact that very niche tracks suited to finer tastes can exist, it is no surprise that having so many clusters can be both a blessing and a curse, as overfitting can be an easy trap to fall into, yet having more clusters may be necessary to address more niche musical interests.
    
For the last stave, we felt that in addition to just use K Means clustering directly it would also be worth exploring what would happen if we use an unsupervised learning algorithm used to obtain cluster centers in conjunction with a supervised classification algorithm. We decided to choose K Means (which was the first model we had implemented) alongside KNN (a clustering algorithm).

The following was our approach:
1. We took the training data and ran K Means on it with a predefined number of clusters determined by the analysis of silhouette scores. This gave us clusters and the data points associated with each of the clusters.
2. Going back to our training data, for each of the samples (songs) we added a new attribute specifying the cluster the song was put into.
3. Now that we had some training data in the form of the songs and the clusters associated with the same, we could run a classification algorithm. 
4. For K nearest neighbors, we passed in the training data (features associated with the song) and the corresponding labels (the cluster the song belonged to). Then, for each of the test samples, we would look at the K nearest neighbors and do a majority vote to determine which class the test sample belonged to. 


We feel that using an approach like KNN would have made our model more robust to outliers because we would look at several similar samples in our feature space close to a given point before arriving at a prediction for that point.  


# Conclusion

We believe that the future of this project would be to gather some real training data from different people, specifically asking if their playlist satisfied their requests. This will be helpful to improve our model performance.  This ties into the idea that, once again, real validation would be extremely helpful in evaluating user preferences, and allow us to perform guided supervised learning models.  Given this data, we may have trained a neural network and used gradient descent to train specifically for weights based on the parameters of user preferences.

Furthermore, to increase the range and variety of songs that our playlist customizer enables, it would have been nice if we were able to scrape songs from the internet and generate the metrics like danceability, acousticness, etc. for each song. Finding a way to automate this so that it is done continuously would drastically increase the robustness and applicability of our model. 

Lastly, if we were to build further on the project, we would try to integrate this model into a user application. Currently, we take in user preferences for one song via a text input, and the model spits out a list of songs that the user may enjoy. Perhaps, in the future we could build an application that allows the user to “graph” each metric over time in a UI as well as specify some other parameters like playlist length, and have the model generate a couple of playlists that adhere to the user’s specifications. Combined with the aforementioned web scraping of songs, this application could potentially be a useful tool for users to generate fine-tuned playlists to their liking.


# Collaboration

- Steven Shi: Worked on the data preprocessing and framework for subsequent models.
- Arnav Modi: Worked on the training and evalutation of the first model involving K Means  and silhouette score analysis. Contributed to the KNN for model 3 as well.
- Yuxuan Wu: Worked on data visualization for EDA, developings figures and analyzing them, notebook writeups for models.
- Bryan Zhu: Worked on development of Models 2 DBBSCAN and 3 KNN, evaluating model performance and hyperparameter tuning.  Helped on Write-Up.
- Anya Chernova: Worked on data visuzalization and exploration, contributed to KNN Model, helped on Write-Up
- Kenny Qiu: Worked on Exploratory Data Analysis and write-ups for milestones.
- Alexander Zhang: Worked on the data preprocessing and compilation of results.
