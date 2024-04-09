import time
import os
import json
import copy
from indexes_enum import Indexes
from preprocess import Preprocessor

class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
            Indexes.DOCTERMS.value: [],
        }
        self.index['docterms'].extend([key.lower() for key in self.index['stars'].keys()])
        self.index['docterms'].extend([key.lower() for key in self.index['genres'].keys()])
        self.index['docterms'].extend([key.lower() for key in self.index['summaries'].keys()])
        
        # self.preprocess()
    
    # def preprocess(self):
    #     """
    #     Preprocess the text using the methods in the class.

    #     Returns
    #     ----------
    #     List[str]
    #         The preprocessed documents.
    #     """
    #      # TODO
        
    #     for doc in self.preprocessed_documents:
    #         for field in doc:
    #             if isinstance(doc[field], str):
    #                 self.index['docterms'].extend(doc[field].split())
    #             elif isinstance(doc[field], list):
    #                 self.step_list_preprocess(doc[field])
            
            
        
    # def step_list_preprocess(self, list_field): 
    #     if not list_field: 
    #         return None
        
    #     new_list_field= []

    #     if isinstance(list_field[0], str):
    #         for a in list_field:
    #             self.index['docterms'].extend(a.split())
    #         return new_list_field
        
    #     for a in list_field:
    #         new_list_field.append(self.step_list_preprocess(a))
    #     return new_list_field

        
              
    def get_tf(self, doc, term):
        tf= 0
        # self.index['docterms'].append(term)
        for field in doc:
            if isinstance(field, str):
                tf+= field.count(term)
            else:
                if field:
                    if isinstance(field[0], str):
                        for string_val in field:
                            tf+= string_val.count(term)
                    else:
                        for list_val in field:
                            for string_val in list_val:
                                tf+= string_val.count(term)
        return tf
        

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """
        # TODO
        current_index = {}
        for doc in self.preprocessed_documents:
            current_index[doc['id']]= doc 

        return current_index

    
    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        #         TODO

        star_index= {}
        seen_names= set()
        
        for doc in self.preprocessed_documents:
            stars= doc['stars']
            # names= [star.split() for star in stars]
            # words= [name for name in names.split()]
            for star in stars:
                for name in star.split():
                    if name not in seen_names:
                        seen_names.add(name)
                        star_index[name]= {}
                        star_index[name][doc['id']]= ' '.join(stars).count(name) #self.get_tf(doc, star)
                    else:  
                        star_index[name][doc['id']]= ' '.join(stars).count(name) #self.get_tf(doc, star)
        return star_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        #         TODO
        
        genre_index= {}
        seen_genres= set()
        
        for doc in self.preprocessed_documents:
            genres= doc['genres']
            for genre in genres:
                if genre not in seen_genres:
                    seen_genres.add(genre)
                    genre_index[genre]= {}
                    genre_index[genre][doc['id']]= 1#self.get_tf(doc, genre)
                else:
                    genre_index[genre][doc['id']]= 1 #self.get_tf(doc, genre)

        return genre_index
    
    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        
        #         TODO
        summary_index= {}
        seen_terms= set()
        
        for doc in self.preprocessed_documents:
            doc_id= doc['id']
            freqs= {}
            summaries= doc['summaries']
            if summaries:
                for summary in summaries:
                    terms= summary.split()
                    for term in terms:
                        if term not in seen_terms:
                            seen_terms.add(term)
                            summary_index[term]= {}
                            summary_index[term][doc_id]= summary_index[term].get(doc_id, 0)+1 #self.get_tf(doc, term)
                        else:
                            summary_index[term][doc_id]= summary_index[term].get(doc_id, 0)+1 #self.get_tf(doc, term)

        return summary_index
       

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        
        try:
            # print(len(self.index[index_type]))
            #         TODO
            index_dict= self.index[index_type]
            ids= index_dict.get(word, {}).keys()
            # ids= []
            # for key, value in index_dict.items():
            #     if word in key:
            #         ids.extend(list(value.keys()))

            return list(ids)
         
        except:
            return []


    # def get_term_index(self, index_type, doc):
    
    #     index_dict= self.index[index_type]
    #     if index_type!='summaries':
    #         if doc[index_type]:
    #             for term in doc[index]
            
    #     for term in index_dict:
    #         if index_type=='stars':  
    #             tf= doc[index_type].count(term)
    #         elif index_type=='genres':
    #             tf= int(term in doc[index_type])
    #         elif index_type=='summaries':
    #             tf= 
                
    #         if tf!=0:
    #             index_dict[term][doc['id']]= tf
    #     return index_dict
                                  
                                  
    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """

        #         TODO

        self.index['documents'][document['id']]= document
        
        stars= document['stars']
        if stars:
            for star in stars:
                for name in star.split():
                    if name not in self.index['stars']:
                        self.index['stars'][name]= {}
                        self.index['stars'][name][document['id']]= star.count(name)
                    else:  
                        self.index['stars'][name][document['id']]= star.count(name)
         
     
        genres= document['genres']
        if genres:
            for genre in genres:
                if genre not in self.index['genres']:
                    self.index['genres'][genre]= {}
                    self.index['genres'][genre][document['id']]= 1
                else:
                    self.index['genres'][genre][document['id']]= 1 

        
        doc_id= document['id']
        freqs= {}
        summaries= document['summaries']
        if summaries:
            for summary in summaries:
                terms= summary.split()
                for term in terms:
                    if term not in self.index['summaries']:
                        self.index['summaries'][term]= {}
                        self.index['summaries'][term][doc_id]= self.index['summaries'][term].get(doc_id, 0)+1
                    else:
                        self.index['summaries'][term][doc_id]= self.index['summaries'][term].get(doc_id, 0)+1 


    def remove_doc_id(self, index_type, doc_id):
        
        index_dict= self.index[index_type]
        
        for term in index_dict:
            if doc_id in index_dict[term]:
                del index_dict[term][doc_id]
        return index_dict
                            
    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        if document_id in self.index['documents']:
            del self.index['documents'][document_id]
        self.index['stars']= self.remove_doc_id('stars', document_id)
        self.index['genres']= self.remove_doc_id('genres', document_id)
        self.index['summaries']= self.remove_doc_id('summaries', document_id)


    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

       
        
        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        # TODO
        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)
        

        if not set(index_after_add).difference(set(index_before_add)):
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
         index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name not in self.index:
            raise ValueError('Invalid index name')
        
        
        # TODO
        
        with open(os.path.join(path, index_name+'_index.json'), 'w') as f:
            json.dump((self.index[index_name]), f)
        

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """

        #         TODO
       
        with open(path+'_index.json', 'r') as f:
            return (json.load(f)) 


    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """
        print(self.index[index_type]==loaded_index)

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'no'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            
            if index_type not in document or document[index_type] is None:
                continue

            
            for field in document[index_type]:
                # it was wrong before. we don't want the ones with the word in them. we want the index. so it has to be equal, not in 
                if check_word == field:
                    # print(document[index_type])
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        
        end = time.time()
        implemented_time = end - start
        # for doc in posting_list:
        #     print(self.preprocessed_documents['id'==doc])
            
        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

# TODO: Run the class with needed parameters, then run check methods and finally report the results of check methods

# with open('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/IMDB_crawled.json', 'r') as f:
#     documents= json.load(f)

# preprocessor= Preprocessor(documents[0])
# processed_docs= preprocessor.preprocess()

# with open('preprocessed_documents.json', 'w') as f:
#     json.dump(list(processed_docs), f)
# with open('preprocessed_documents.json', 'r') as f:
#     processed_documents= json.load(f)

with open('IMDB_crawled.json', 'r') as f:
    processed_documents= json.load(f)
 
created_index= Index(processed_documents)
# created_index.check_if_index_loaded_correctly('documents', created_index.load_index('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/indexer/index/documents'))
# created_index.check_if_index_loaded_correctly('stars', created_index.load_index('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/indexer/index/stars'))
# created_index.check_if_index_loaded_correctly('genres', created_index.load_index('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/indexer/index/genres'))
# created_index.check_if_index_loaded_correctly('summaries', created_index.load_index('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/indexer/index/summaries'))

# created_index.check_add_remove_is_correct()

created_index.check_if_indexing_is_good('stars')
created_index.check_if_indexing_is_good('genres')
created_index.check_if_indexing_is_good('summaries')

# created_index.store_index('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/indexer/index/', 'documents')
# created_index.store_index('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/indexer/index/', 'stars')
# created_index.store_index('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/indexer/index/', 'genres')
# created_index.store_index('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/indexer/index/', 'summaries')
# created_index.store_index('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/indexer/index/', 'docterms')
