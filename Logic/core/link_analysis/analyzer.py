import sys
import os

current= os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(current))))


from core.link_analysis.graph import LinkGraph
from core.indexer.indexes_enum import Indexes
from core.indexer.index_reader import Index_reader
import numpy as np
import json

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.titles= {}
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            self.graph.add_node(movie['id'])
            self.hubs.append(movie['id'])
            self.titles[movie['id']]= movie['title']
            if movie['stars']:
                for star in movie['stars']:
                    self.graph.add_node(star)
                    self.authorities.append(star)
                    self.graph.add_edge(movie['id'], star)

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            for star in self.authorities:
                if movie['stars']:
                    if star in movie['stars']:
                        if movie['id'] not in self.hubs:
                            self.hubs.append(movie['id'])
                            self.titles[movie['id']]= movie['title']
                            self.graph.add_node(movie['id'])
                            self.graph.add_edge(movie['id'], star)
                            
                            if movie['stars']:
                                for other_star in movie['stars']:
                                    if star!=other_star:
                                        if other_star not in self.authorities:
                                            self.authorities.append(other_star)
                                            self.graph.add_node(other_star)
                                            self.graph.add_edge(movie['id'], other_star)
                                        else:
                                            self.graph.add_edge(movie['id'], other_star)
                        break
                
    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s= np.ones(len(self.authorities))
        h_s= np.ones(len(self.hubs))

        for i in range(num_iteration):
            
            for i, h in enumerate(self.hubs):
                a_succ= self.graph.get_successors(h)
                for succ in a_succ:
                    h_s[i]+= a_s[self.authorities.index(succ)]
                
            for i, a in enumerate(self.authorities):
                h_pred= self.graph.get_predecessors(a)
                for pred in h_pred:
                    a_s[i]+= h_s[self.hubs.index(pred)]

            a_s_sum= sum(a_s)
            h_s_sum= sum(h_s)
            
            for i in range(len(a_s)):
                a_s[i]/= a_s_sum
                
            for i in range(len(h_s)):
                h_s[i]/= h_s_sum
            
        a_s, a_res= zip(*sorted(zip(a_s, self.authorities)))
        h_s, h_res= zip(*sorted(zip(h_s, self.hubs)))
        # h_res= [self.titles[m_id] for m_id in h_res]
        
        return a_res[:max_result], h_res[:max_result]


# You can use this section to run and test the results of your link analyzer
if __name__ == "__main__":
    corpus = []    # it shoud be your crawled data
        
    with open('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/IMDB_crawled.json', 'r') as f:
        corpus= json.load(f)
            
    # we pretend we want to check among the movies with the word spider in them    

    root_set = []   # it shoud be a subset of your corpus
    for movie in corpus:
        if 'spider' in movie['title'].lower():
            root_set.append(movie)
                
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
