import numpy as np
import itertools
import random
import json
import secrets



class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.testdocs= None
        
        self.documents = documents
        self.num_hashes = num_hashes
        self.all_shingles= set()
        self.doc_shingles= list()

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        
        shingles= set()
        
        words= document.split()
        for i in range(len(words)-k+1):
            shingle= ' '.join(words[i:i+k])
            shingles.add(shingle)
            if shingle not in self.all_shingles:
                self.all_shingles.add(shingle)
        
        return shingles

        # shingles = []
        # words = document.split()
        # for i in range(len(words) - k + 1):
        #     shingle = ()
        #     for j in range(k):
        #         shingle = shingle + (words[i + j],)
        #     shingles.append(shingle)
        # shingles = set(shingles)
        
        # return shingles



    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        # TODO
        
        docs_shingles= []  
        for doc in self.documents:
            docs_shingles.append(self.shingle_document(doc))
        self.doc_shingles= docs_shingles
            
        # ch_matrix= np.zeros([len(self.all_shingles), len(self.documents)])
        # for i, shingle in enumerate(self.all_shingles):
        #     for j, doc in enumerate(self.documents):
        #         if shingle in doc:
        #             ch_matrix[i, j]= 1
        
        # return ch_matrix 

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        # TODO

        max_range= 2**32-1
        seeds= []
        for i in range(self.num_hashes):
            seeds.append(secrets.randbelow(max_range))
        sign_matrix = np.full((self.num_hashes, len(self.documents)), np.inf)
        char_matrix= self.build_characteristic_matrix()

        print(seeds[0])
        for j, shingles in enumerate(self.doc_shingles):
            for i, seed in enumerate(seeds):
                for shingle in shingles:
                    hash_val= seed
                    for char in ' '.join(shingle):
                        hash_val= (hash_val*10+ord(char))%(1<<32)
                    sign_matrix[i][j]= min(sign_matrix[i][j], hash_val)
        return sign_matrix
     
    
        # why didn't the slides work?
        # char_matrix= self.build_characteristic_matrix()
        # hash_function= list(range(char_matrix.shape[0]))
        # sign_matrix= np.zeros([self.num_hashes, len(self.documents)])

        # for i in range(self.num_hashes):
        #     random.shuffle(hash_function)
        #     for j in range(len(self.documents)):
        #         vals= np.multiply(hash_function, char_matrix[:, j])
                
        #         if any(vals[vals!=0]):
        #             sign_matrix[i, j]= min(vals[vals!=0])
        #         else:
        #             sign_matrix[i, j]= 0
        # return sign_matrix

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        buckets= {}
        for i in range(bands):
            for j in range(len(self.documents)):
                bucket_hash= hash(tuple(signature[i*rows_per_band:(i+1)*rows_per_band, j]))
                if bucket_hash in buckets:
                    buckets[bucket_hash].append(j)
                else:
                    buckets[bucket_hash]= [j]   
           
        # count= 0         
        # one_count=0
        # for bucket in buckets.values():
        #     if len(bucket)>0:
        #         count+= 1
        #         if len(bucket)==1:
        #             one_count+=1 
        # print("number of non zero buckets= ", count)
        # print("number of one buckets= ", one_count)
        # print("number of all buckets: ", len(buckets))
        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        
        return self.lsh_buckets(self.min_hash_signature())

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        # TODO
        
        if len(second_set)<len(first_set):
            temp_set= first_set.copy()
            first_set= second_set
            second_set= temp_set
        
        intersect_count= 0
        
        for shingle in first_set:
            if shingle in second_set:
                intersect_count+= 1
        
        union_len= len(first_set.union(second_set))
        if union_len==0:
            return 0.0
        return (intersect_count/union_len)

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)




with open('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/LSHFakeData.json', 'r') as f:
    movies= json.load(f)

docs= [' '.join(movie['summaries']) for movie in movies]
with open('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/IMDB_crawled.json', 'r') as f:
    movies= json.load(f)
docs.extend([' '.join(movie['summaries']) for movie in movies if len(movie['summaries'])>0])
print(len(docs))

minHashLSH = MinHashLSH(docs, 100)
minHashLSH.jaccard_similarity_test(minHashLSH.perform_lsh(), docs)
