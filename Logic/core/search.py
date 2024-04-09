import json
import numpy as np
import sys
sys.path.append("../Logic/core/")

from preprocess import Preprocessor
from scorer import Scorer
from indexer.indexes_enum import Indexes, Index_types
from indexer.index_reader import Index_reader


class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        path = 'C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/indexer/index/'
        self.document_indexes = {
            Indexes.DOCUMENTS.value: Index_reader(path, Indexes.DOCUMENTS, None).index,
            Indexes.STARS.value: Index_reader(path, Indexes.STARS, None).index,
            Indexes.GENRES.value: Index_reader(path, Indexes.GENRES, None).index,
            Indexes.SUMMARIES.value: Index_reader(path, Indexes.SUMMARIES, None).index
        }
        self.tiered_index = {
            Indexes.STARS.value: Index_reader(path, Indexes.STARS, Index_types.TIERED).index,
            Indexes.GENRES.value: Index_reader(path, Indexes.GENRES, Index_types.TIERED).index,
            Indexes.SUMMARIES.value: Index_reader(path, Indexes.SUMMARIES, Index_types.TIERED).index
        }
        self.document_lengths_index = {
            Indexes.STARS.value: Index_reader(path, Indexes.STARS, Index_types.DOCUMENT_LENGTH).index,
            Indexes.GENRES.value: Index_reader(path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH).index,
            Indexes.SUMMARIES.value: Index_reader(path, Indexes.SUMMARIES, Index_types.DOCUMENT_LENGTH).index
        }
        self.metadata_index = Index_reader(path, Indexes.DOCUMENTS, Index_types.METADATA).index
        self.num_of_documents= len(self.document_indexes[Indexes.DOCUMENTS.value].keys())
        

    def search(self, query, method, weights, safe_ranking, max_results=10):
        """
        searches for the query in the indexes.

        Parameters
        ----------
        query : str
            The query to search for.
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        safe_ranking : bool
            If True, the search engine will search in whole index and then rank the results. 
            If False, the search engine will search in tiered index.
        max_results : int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            A list of tuples containing the document IDs and their scores sorted by their scores.
        """

        preprocessor = Preprocessor([query])
        query = preprocessor.preprocess()[0].split()

        scores = {}
        if safe_ranking:
            self.find_scores_with_safe_ranking(query, method, weights, scores)
        else:
            self.find_scores_with_unsafe_ranking(query, method, weights, max_results, scores)

        final_scores = {}

        self.aggregate_scores(weights, scores, final_scores)
        
        result = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        if max_results is not None:
            result = result[:max_results]

        return result

    def aggregate_scores(self, weights, scores, final_scores):
        """
        Aggregates the scores of the fields.

        Parameters
        ----------
        weights : dict
            The weights of the fields.
        scores : dict
            The scores of the fields.
        final_scores : dict
            The final scores of the documents.
        """
        
        # TODO
      
        for field in weights:
            if field in scores:
                for doc_id in scores[field]:
                    if doc_id not in final_scores:
                        final_scores[doc_id]= (weights[field]*scores[field][doc_id])
                    else:
                        final_scores[doc_id]+= (weights[field]*scores[field][doc_id])
        
        
        return final_scores

    def find_scores_with_unsafe_ranking(self, query, method, weights, max_results, scores):
        """
        Finds the scores of the documents using the unsafe ranking method using the tiered index.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        max_results : int
            The maximum number of results to return.
        scores : dict
            The scores of the documents.
        """
        

        for field in weights:
            
            # field= None
            # if field_s=='stars':
            #     field= Indexes.STARS
            # elif field_s=='genres':
            #     field= Indexes.GENRES
            # else:
            #     field= Indexes.SUMMARIES
            tier_scores= {}
            scores[field]= {}
            till_now= set()
            
            for tier in ["first_tier", "second_tier", "third_tier"]:
                scorer= Scorer(self.tiered_index[field][tier], self.num_of_documents)
                if method=='OkapiBM25':
                    scores[field]= scorer.compute_socres_with_okapi_bm25(query, float(self.metadata_index['averge_document_length'][field]), self.document_lengths_index[field]) 
                else:
                    scores[field]= scorer.compute_scores_with_vector_space_model(query, method)    
                
                tier_scores[tier]= scores[field]
                
                if len(till_now)>max_results:
                    break
                
                for key in scores[field].keys():
                    till_now.add(key)

                
            while len(tier_scores.keys())>1:
                # print(tier_scores)
                first= list(tier_scores.keys())[0]
                second= list(tier_scores.keys())[1]
                merged_scores= self.merge_scores(tier_scores[first], tier_scores[second])
                
                del tier_scores[first]
                del tier_scores[second]
                
                tier_scores[first+second]= merged_scores
            
            scores[field]= tier_scores[list(tier_scores.keys())[0]]
    
    def find_scores_with_safe_ranking(self, query, method, weights, scores):
        """
        Finds the scores of the documents using the safe ranking method.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        scores : dict
            The scores of the documents.
        """
        
        for field in weights.keys():
            #TODO
            # field= None
            # if field_s=='stars':
            #     field= Indexes.STARS
            # elif field_s=='genres':
            #     field= Indexes.GENRES
            # else:
            #     field= Indexes.SUMMARIES
                
            scorer= Scorer(self.document_indexes[field], self.num_of_documents)
            scores[field]= {}
            if method=='OkapiBM25':
                scores[field]= scorer.compute_socres_with_okapi_bm25(query, float(self.metadata_index['averge_document_length'][field]), self.document_lengths_index[field]) 
            else:
                scores[field]= scorer.compute_scores_with_vector_space_model(query, method)    
       


    def merge_scores(self, scores1, scores2):
        """
        Merges two dictionaries of scores.

        Parameters
        ----------
        scores1 : dict
            The first dictionary of scores.
        scores2 : dict
            The second dictionary of scores.

        Returns
        -------
        dict
            The merged dictionary of scores.
        """
        merged_dict={}
        
        first_keys= set(scores1.keys())
        second_keys= set(scores2.keys())
        
        for key in list(first_keys.union(second_keys)):
            merged_dict[key]= 0
            if key in scores1:
                merged_dict[key]+= scores1[key]
            if key in scores2:
                merged_dict[key]+= scores2[key]
        
        return merged_dict
                
        #TODO


if __name__ == '__main__':
    search_engine = SearchEngine()
    query = "spiderman"
    method = "OkapiBM25"
    weights = {
        Indexes.STARS.value: 1,
        Indexes.GENRES.value: 1,
        Indexes.SUMMARIES.value: 1
    }
    result = search_engine.search(query, method, weights, False)
    print(weights)
    print(result)
    for res in result:
        print(search_engine.document_indexes[Indexes.DOCUMENTS.value][res[0]]['title'])