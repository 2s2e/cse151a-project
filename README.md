# Introduction

The purpose of our machine learning application is to allow the user to create very specific playlists that suit their needs. We aim to build a recommender system that first learns the user’s preferences and song associations and then allows them to provide a precise description of a playlist they’d like for the model to generate. For example, a user could ask for a playlist that contains a large amount of lyrics, starts slow, but has faster songs as the playlist progresses. We foresee the use of both unsupervised learning, for grouping similar songs, and supervised learning, for adjusting to the user’s preferences. 

We were interested in this idea because we wanted to offer the users a different and more precise personalized experience. Unlike standard recommenders that generate playlists based on your recently played songs which often lack nuance and flexibility in their algorithm, our application empowers users to curate playlists that resonate deeply with their unique tastes and moods. On top of that, our application also encourages users’ curiosity by allowing them to fine-tune the variables, thus allowing them to explore a wider range of music that they would not have discovered otherwise. Furthermore, by introducing users to a wider variety of music, our application can promote diversity within the music industry and support lesser-known artists by exposing their work to new audiences. 


# Method
### Data Exploration
```
df_HU = df[df['Artist'] == 'Hollywood Undead']
num_cols = ["Danceability", "Energy", "Loudness", "Speechiness", "Acousticness",
            "Instrumentalness", "Liveness", "Valence", "Tempo", "Duration_ms",
            "Views", "Likes", "Comments", "Stream"]
df_HU_num = df_HU[num_cols]
plt.figure(figsize=(10, 8))
sns.heatmap(df_HU_num.corr(), vmin=-1, vmax=1, center= 0, linewidth=2, fmt='.2g',
            cmap= 'coolwarm', annot=True)
plt.show()
```
![image](https://github.com/2s2e/cse151a-project/assets/97645823/74fc0c0b-a43b-4b42-98b4-4d1405668bed)

**Data Exploration Fig 1. Correlation Heatmap of Musical Features**

1. There is a somewhat substantial positive correlation between "danceability" and "valence," indicating that songs that are deemed more danceable are typically linked to upbeat or joyful music.
2. There is a strong positive link between "energy" and "loudness," which supports the idea that louder music is frequently interpreted as being more lively.
3. "Acousticness" and "Energy" have a substantial negative association, suggesting that songs with more acoustic material are typically less energetic.
4. There is a high correlation between "Stream," "Likes," "Views," and "Comments," indicating that popular music typically do well across these measures.
5. There is a negative link between "Duration_ms" and "Danceability," "Energy," and "Valence," suggesting that danceable and energetic tunes are better off with shorter songs.


![image](https://github.com/2s2e/cse151a-project/assets/97645823/cdc6b64e-87c3-4efa-8ec8-be69cd02a07c)

```
logarithmic_categories = ["Duration_ms", "Views", "Likes", "Comments", "Stream"]

# Calculate the number of rows and columns for the subplot grid
n_cols = 3
n_rows = len(logarithmic_categories) // n_cols + (1 if len(logarithmic_categories) % n_cols > 0 else 0)

# Create a figure and a grid of subplots
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(19, 10))
axs = axs.flatten()

for idx, column in enumerate(logarithmic_categories):
    # Apply a logarithmic transformation with a small shift to handle zero values
    logarithmic_data = np.log(df[column] + 1)
    axs[idx].hist(logarithmic_data, bins=50, color='blue', alpha=0.7)
    axs[idx].set_title(column)
    axs[idx].set_xlabel('Logarithmic Scale')
    axs[idx].set_ylabel('Frequency')

# Hide any empty subplots
for ax in axs[len(logarithmic_categories):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()
```

**Data Exploration Fig 2. Histograms for Log-Transformed Music Track Features**

This is a collection of histograms to display the frequency distributions of five key features from a music track dataset—'Duration_ms', 'Views', 'Likes', 'Comments', and 'Stream' after applying a logarithmic transformation to each. 

Most of them show standard deviation, 'Views' and 'Likes' are slightly skewed to the right.

![image](https://github.com/2s2e/cse151a-project/assets/97645823/f3e5d24e-a3dd-4ac3-b189-a8ab1a07a8b1)

```
df_only_categorical['Album_type'].value_counts()

fig = plt.figure(figsize=(10, 7))
df_only_categorical['Album_type'].value_counts().plot(kind='pie', autopct='%0.1f%%',shadow=True,explode=(0.1,0.1,0.1),wedgeprops={'linewidth':2,'edgecolor':'black'})

plt.show()
```

**Data Exploration Fig 3. Pie Chart Distribution of Album Types in Music Dataset**

Distribution of album types to observe what kind of albums will be analyzed within the dataset.  We observe large proportion of album as a type, while singles follow, and a sliver of compilations.

The percentage of each album type—album, single, and compilation—in a music dataset is shown in a pie chart. With albums making up the greatest portion of the dataset (72% of the total), it is likely that typical full-length albums make up the majority of the dataset. Singles make up 24.2% of the population, a substantial but relatively tiny representation. With only 3.8% of the dataset, compilations are the least common type of data. 

###    Preprocessing

We first dropped irrelevant features such as Spotify/Youtube URLs, Channel, whether or not they are Licensed, and whether or not they are official video.

```
def drop_irrelevant_cols(df, irrelevant_columns):

    input_num_cols = df.shape[1]
    print(f"The original dataset has {input_num_cols} columns", end = "\n\n")

    df = df.drop(columns=irrelevant_columns)
    print(f"{len(irrelevant_columns)} columns were successfully dropped from the dataframe!")
    print(irrelevant_columns, end = "\n\n")

    output_num_cols = df.shape[1]
    print(f"The final dataset has {output_num_cols} columns", end = "\n\n")

    return df
```

```
irrelevant_columns = ['Url_youtube', 'Url_spotify', 'Uri', 'Unnamed: 0', 'Channel', 'Licensed', 'official_video']
df = drop_irrelevant_cols(df, irrelevant_columns)
df.head()
```

We first dropped irrelevant features such as Spotify/Youtube URLs, Channel, whether or not they are Licensed, and whether or not they are official videos.

![image](https://github.com/2s2e/cse151a-project/assets/97645823/57d596bd-06ab-4065-b3f4-0fb52c9af2d5)

Then we dropped the rows that have null values.

```
def drop_null_values(df):

    initial_num_rows = df.shape[0]
    print(f"Number of rows in the dataframe before dropping nulls: {initial_num_rows}", end = "\n\n")

    df.dropna(inplace=True)
    print(f"Null values were dropped from the dataframe!", end = "\n\n")

    output_num_rows = df.shape[0]
    print(f"Number of rows in the dataframe after dropping nulls: {output_num_rows}", end = "\n\n")

    return df
```

![image](https://github.com/2s2e/cse151a-project/assets/97645823/790ebed1-fd67-4ab3-9ce6-6e16e9ace018)


Then we one-hot encoded the Key of the Song and Album Type (Single, Album, Compilation). 

```
def oneHotEncodeFeatures(df, feature_list):

    print("Features we are encoding:")
    print(feature_list, end = "\n\n")
    encoder = OneHotEncoder()

    print(f"Number of columns before one hot encoding {df.shape[1]}", end = "\n\n")

    encoded_df = pd.DataFrame(encoder.fit_transform(df[feature_list]).toarray())
    print("One hot encoding completed!", end = "\n\n")

    df.drop(columns=feature_list)

    df = df.join(encoded_df)

    print(f"Number of columns before one hot encoding {df.shape[1]}", end = "\n\n")

    return df
```

We have logarithmically scaled the data, which resulted in a normal distribution, and then normalized the data to be between 0 and 1. Strings, such as artist name and album name were encoded based on the total number of views associated with that artist or album. 

![image](https://github.com/2s2e/cse151a-project/assets/97645823/5e3c90f1-fad3-4258-bc6a-c23391d2bdf6)

### Model 1

![image](https://github.com/2s2e/cse151a-project/assets/97645823/bb76dd22-4002-4db7-abf5-9d65ed27c13f)

**Model 1 Fig 1. Silhouette Analysis for Optimal Cluster Number Determination in KMeans Clustering**

The silhouette plot shows the degree to which each object has been correctly classified using KMeans with 25 clusters during the clustering process. The silhouette coefficient, which gauges a sample's similarity to samples in its own cluster relative to samples in other clusters, is represented by each bar for a sample within a cluster. The coefficient ranges from -1 to 1,-+ where a high value indicates that the sample is well matched to its own cluster and poorly matched to neighboring clusters.


```python=
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

for n_clusters in range(5, 30, 5):
    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(5, 10)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(Xtrain_k) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, n_init='auto' ,random_state=10)
    cluster_labels = clusterer.fit_predict(Xtrain_k)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(Xtrain_k, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(Xtrain_k, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )


n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(Xtrain_k)
Xtrain_with_clusters = Xtrain_k
Xtrain_with_clusters['cluster'] = kmeans.labels_
```

### Model 2

We chose DBSCAN as our second model for clustering. To perform a cluster analysis, we used the numerical data of Danceability, Energy, Loudness, Speechiness, and Acousticness (which we are planning to accept as user input parameters in our application). We used the same data we trained our first model on in order to compare the performance of the two models and choose the best one for the next steps. Like for the last clustering algorithm, given the lack of labels to verify the grouping, we ended up using metrics dedicated for clustering algorithms evaluation, specifically silhouette score, and Within Cluster Sum of Squares, to assess model performance on training and test data.
We performed hyper-parameter tuning on our DBSCAN model. This was done by GridSearch: we iteratively created models with different epsilon values and min_samples in certain range. We track the parameters that produce the clusters with the greatest silhouette score as our metric, suggesting the best fit clusters. As such we were able to choose a more optimized clustering model, which had a score of approximately 0.56, which is a considerable performance increase than other hyperparameter configurations that could go as low as 0.30.

![image](https://github.com/2s2e/cse151a-project/assets/97645823/eadfec3c-4762-4f4a-ad03-7beb57df7a43)

**Model 2 Fig 1. Histogram on Feature Distributions within Cluster 0**

1. The 'Danceability' histogram shows a distribution with a concentration around the 0.7 to 0.8 range, suggesting that most tracks in this cluster are quite danceable.
2. 'Energy' seems to be broadly distributed with multiple peaks indicating variability in the energy level, but with the majority of songs concentrated in the range of 0.7 - 0.9.
3. 'Loudness' exhibits a peak around 0.875, implying that tracks in cluster 0 tend to be loud.
4. 'Speechiness' shows a left-skewed distribution, with most tracks having lower speechiness values.
5. The 'Acousticness' histogram is heavily left-skewed which reveals a preference for lower acousticness in this cluster.

```python=
from sklearn.metrics import silhouette_score

# Iteratively search through hyperparameters epsilon and min samples
for i in range(MIN_SAMPLES_START, MIN_SAMPLES_END + 1, SAMPLE_STEP):
    for j in EPS_VALUES:
        clustering = DBSCAN(eps=j, min_samples=i).fit(Xtrain)
        Xtrain['cluster'] = clustering.labels_
        Xtrain['cluster'].value_counts()

        # Evaluate the clusters via silhouette score
        score = silhouette_score(Xtrain.drop(columns=['cluster']), Xtrain['cluster'])
        print("Silhouette Score for eps = " + str(j) + ", min_samples = " + str(i) + ": " + str(score))

        # If these hyperparameters produce a better score than we last checked for, save it!
        if abs(score) > best_score:
            best_score = abs(score)

            # Save hyperparameters
            best_eps = j
            best_min_samples = i

            print("New optimal hyperparameters found!")

# fit returns a fitted instance of self
model = DBSCAN(eps=best_eps, min_samples=best_min_samples)
clustering = model.fit(Xtrain)
# we will now assign the clusters corresponding to the best set of hyperparameters found
Xtrain['cluster'] = clustering.labels_
```

### Model 3

```python=
def knn_predict(k, clustered_data, clusters, data_to_be_classified):
    """

        Parameters:
            1) k: number of neighbors
            2) clustered_data: data for which clusters are known
            3) clusters: clusters associated with each row of clustered_data
            4) data_to_be_classified: data for which we need to obtain the clusters
    """
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(clustered_data, clusters)
    yhat = neigh.predict(data_to_be_classified)
    return yhat
```

**= WCSS =**
1. From the graph we can make the observation that the WSCC score gets lower as we increased the score we choose for K, but we are seeing a spike as we approach K values greater than 80. We can reason and conclude from the graph that the K value of 60 is optimal.
**= WCSS =**
1. We can see a spike up in WSCC value when our choice of K increased from 0 to 20, and the WCSS score plateaued as K increased from 20 to 70. Finally WCSS score started rising as K approaches 80. However, it's worth noting that the variations in terms of their WCSS values is almost negligible. 


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
