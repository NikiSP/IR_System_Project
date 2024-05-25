import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
import wandb

# from ..word_embedding.fasttext_data_loader import FastTextDataLoader
# from ..word_embedding.fasttext_model import FastText
# from .dimension_reduction import DimensionReduction
# from .clustering_metrics import ClusteringMetrics
# from .clustering_utils import ClusteringUtils

sys.path.append("../")
current= os.path.dirname(__file__)
current= os.path.join(current, '..')
sys.path.append(os.path.dirname((os.path.abspath(current))))


from core.word_embedding.fasttext_data_loader import FastTextDataLoader
from core.word_embedding.fasttext_model import FastText
from dimension_reduction import DimensionReduction
from clustering_metrics import ClusteringMetrics
from clustering_utils import ClusteringUtils



def clustering_print(model_type, clustering_metrics, input_labels, y, reduced_embeddings):
    print(f'{model_type}:')
    print("Silhouette Score:", clustering_metrics.silhouette_score(reduced_embeddings, input_labels))
    print("Purity Score:", clustering_metrics.purity_score(y, input_labels))
    print("Adjusted Rand Score:", clustering_metrics.adjusted_rand_score(y, input_labels), end='\n\n')
    

# Main Function: Clustering Tasks

# 0. Embedding Extraction
# TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.



x= np.load('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/word_embedding/embedding_files/x_clustering.npy')
y= np.load('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/word_embedding/embedding_files/y_clustering.npy')
k_vals= [i for i in range(2,10)]
fasttext_model= FastText()
fasttext_model.prepare(None, "load", False, 'C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/word_embedding/model/FastText_clustering_model.bin')
embeddings= []
for val in x:
    embeddings.append(fasttext_model.get_query_embedding(val))
embeddings= np.array(embeddings)
# 1. Dimension Reduction
# TODO: Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.

dimension_reduction= DimensionReduction()
dimension_reduction.wandb_plot_explained_variance_by_components(embeddings, 'Clustering', 'Explained Variance')

# TODO: Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
#     - Use the output vectors from this step to draw the diagram.

dimension_reduction.wandb_plot_2d_tsne(embeddings, 'Clustering', '2D T-SNE')
reduced_embeddings= dimension_reduction.pca_reduce_dimension(embeddings, 2)

# 2. Clustering
## K-Means Clustering
# TODO: Implement the K-means clustering algorithm from scratch.
# TODO: Create document clusters using K-Means.
# TODO: Run the algorithm with several different values of k.
# TODO: For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster.
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
#     - Check the implementation and efficiency of the algorithm in clustering similar documents.
# TODO: Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
# TODO: Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)

clustering_utils= ClusteringUtils()
clustering_metrics= ClusteringMetrics()

run= wandb.init(project='Clustering', name= 'Kmeans Plots')
for k in range(2, 10):
    clustering_utils.visualize_kmeans_clustering_wandb(reduced_embeddings, k, 'Clustering', 'Kmeans Plots')
wandb.finish()

clustering_utils.plot_kmeans_cluster_scores(reduced_embeddings, y, k_vals, 'Clustering', 'Kmeans Scores')
clustering_utils.visualize_elbow_method_wcss(reduced_embeddings, k_vals, 'Clustering', 'Elbow Method WCSS')


# ## Hierarchical Clustering
# # TODO: Perform hierarchical clustering with all different linkage methods.
# # TODO: Visualize the results.

run= wandb.init(project='Clustering', name= 'Hierarchical Dendograms')
clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(reduced_embeddings, 'Clustering', 'single', 'Single Dendrogram')
clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(reduced_embeddings, 'Clustering', 'complete', 'Complete Dendrogram')
clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(reduced_embeddings, 'Clustering', 'average', 'Average Dendrogram')
clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(reduced_embeddings, 'Clustering', 'ward', 'Ward Dendrogram')
wandb.finish()

# 3. Evaluation
# TODO: Using clustering metrics, evaluate how well your clustering method is performing.

n_eval_clusters= 5

centers, k_labels= clustering_utils.cluster_kmeans(reduced_embeddings, n_eval_clusters)
s_labels= clustering_utils.cluster_hierarchical_single(reduced_embeddings, n_eval_clusters)
c_labels= clustering_utils.cluster_hierarchical_complete(reduced_embeddings, n_eval_clusters)
a_labels= clustering_utils.cluster_hierarchical_average(reduced_embeddings, n_eval_clusters)
w_labels= clustering_utils.cluster_hierarchical_ward(reduced_embeddings, n_eval_clusters)


clustering_print('Kmeans Clustering', clustering_metrics, k_labels, y, reduced_embeddings)
clustering_print('Single Hieraarchical', clustering_metrics, s_labels, y, reduced_embeddings)
clustering_print('Complete Hieraarchical', clustering_metrics, c_labels, y, reduced_embeddings)
clustering_print('Average Hieraarchical', clustering_metrics, a_labels, y, reduced_embeddings)
clustering_print('Ward Hieraarchical', clustering_metrics, w_labels, y, reduced_embeddings)