import re

class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        # TODO: remove stop words from the query.
        with open("C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/utility/stopwords.txt", 'r') as file:
            stopwords= [line.strip() for line in file]
            
        tokens= query.split()
        tokens= [tok for tok in tokens if tokens not in stopwords]

        return ' '.join(tokens)
    
    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        # TODO: Extract snippet and the tokens which are not present in the doc.
        low_query= query.lower()

        # add *** to the string 
        
        reformed_query= self.remove_stop_words_from_query(query)
        words= {}
        for word in list(set(reformed_query.split())):
            
            starts= [match.start() for match in re.finditer(word, doc, re.IGNORECASE)]
            ends= [i+len(word) for i in starts]
            
            if len(starts)==0:
                not_exist_words.append(word)
            else:
                words[word]= len(starts)
                
            offset= 0
            for start, end in zip(starts, ends):
                start+= offset
                end+= offset
                doc= doc[0:start]+'***'+doc[start: end]+'***'+doc[end:]
                offset+= 6
            
        
        
        # add ... to string 
        doc= doc.split()
        parts= []
        
        while True:
            found= False
            start= -1
            for i, term in enumerate(doc):
                if '***' in term:
                    start= i
                    found= True
                    break
            if not found:
                final_snippet= ' '.join(parts)
                return final_snippet, not_exist_words
    
            remove_before= True
            before_count= 0
            start_loop= start-self.number_of_words_on_each_side
            for i in range(start-1, start_loop-1, -1):
                if i<0 or '*' in doc[i]:
                    remove_before= False
                    break
                before_count+= 1


            remove_after= True  
            after_count= 0
            end_loop= start+1+self.number_of_words_on_each_side
            for i in range(start+1, end_loop):
                if i>=len(doc) or '*' in doc[i]:         
                    remove_after= False
                    break
                after_count+= 1
                if after_count==self.number_of_words_on_each_side:
                    for j in range(min(end_loop+self.number_of_words_on_each_side,len(doc))):
                        if '*' in doc[j]:
                            remove_after= False
                        
                
            
            
            if remove_before:
                parts.extend('...')
                
            parts.extend(doc[start-before_count:start+after_count+1])
            
            if remove_after:
                parts.extend('...')
            doc= doc[start+after_count+1:]
      
        return parts, not_exist_words

snippet_finder = Snippet()
doc = "When New York is put under siege by Oscorp, it is up to Spider-Man to save the city he swore to protect as well as his loved ones."
query = "spiderman"
final_snippet, not_exist_words = snippet_finder.find_snippet(doc, query)
print("Final Snippet:", final_snippet)
print("Words not found:", not_exist_words)