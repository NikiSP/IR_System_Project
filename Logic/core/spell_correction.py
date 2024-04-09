import numpy as np

class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        # TODO: Create shingle here
        
        
        shingles = set()
        new_word= '$'+word+'$'
        for i in range(len(new_word)-k+1):
            shingle= ''.join(new_word[i:i+k])
            shingles.add(shingle)
            
        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        # TODO: Calculate jaccard score here.
        
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

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()

        # TODO: Create shingled words dictionary and word counter dictionary here.
        
        for doc in all_documents:
            unique_words= set(doc.lower().split())
            for word in unique_words:
                if word not in all_shingled_words:
                    all_shingled_words[word]= self.shingle_word(word)
                    word_counter[word]= doc.count(word)
                word_counter[word]+= doc.count(word)
                
                
        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """  
             
        # TODO: Find 5 nearest candidates here.
        scores= np.zeros(5)
        words= ['', '', '', '', '']
        
        # top5_candidates = list()
        query_shingle= self.shingle_word(word)
        
        for doc_word in self.all_shingled_words:
            score= self.jaccard_score(query_shingle, self.all_shingled_words[doc_word])
            min_index= np.argmin(scores)
            if score>=scores[min_index]:
                scores[min_index]= score
                words[min_index]= doc_word
 
        return list(words), scores
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        # final_result = ""
        
        # TODO: Do spell correction here.
        results= []
        
        for word in query.split():
            candidates, scores= self.find_nearest_words(word)
            tfs= np.array([self.word_counter[cand] for cand in candidates])
            tfs= tfs/max(tfs)
            
            scores= np.multiply(tfs, scores)
            results.append(candidates[np.argmax(scores)])
        
        return ' '.join(results)
    
    