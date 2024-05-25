
centers, k_labels= clustering_utils.cluster_kmeans(reduced_embeddings, 5)
s_labels= clustering_utils.cluster_hierarchical_single(reduced_embeddings)
c_labels= clustering_utils.cluster_hierarchical_complete(reduced_embeddings)
a_labels= clustering_utils.cluster_hierarchical_average(reduced_embeddings)
w_labels= clustering_utils.cluster_hierarchical_ward(reduced_embeddings)


clustering_print('Kmeans Clustering', clustering_metrics, k_labels, y, reduced_embeddings)
clustering_print('Single Hieraarchical', clustering_metrics, s_labels, y, reduced_embeddings)
clustering_print('Complete Hieraarchical', clustering_metrics, c_labels, y, reduced_embeddings)
clustering_print('Average Hieraarchical', clustering_metrics, a_labels, y, reduced_embeddings)
clustering_print('Ward Hieraarchical', clustering_metrics, w_labels, y, reduced_embeddings)