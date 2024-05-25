import fasttext
import tempfile
import re
import sys
import os 
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance

# from .fasttext_data_loader import FastTextDataLoader
sys.path.append('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/word_embedding/')

from fasttext_data_loader import FastTextDataLoader

def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                       punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    
    # we do not need this funciton. all our data has already been preprocessed using the Preprocess module in utility folder
    # indexes have been created using the preprocessed data, so we are good to use the data as it is without furthur preprocessing
    
    
    pass

class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, method='skipgram'):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.method = method
        self.model = None


    def train(self, texts):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        
        input_file= tempfile.NamedTemporaryFile(delete=False)
        with open(input_file.name, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text+'\n')
                
        self.model= fasttext.train_unsupervised(input_file.name, model=self.method)

        input_file.close()
        
    def get_query_embedding(self, query):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        return self.model.get_sentence_vector(query)

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        # Obtain word embeddings for the words in the analogy
        word1_v= self.model[word1]
        word2_v= self.model[word2]
        word3_v= self.model[word3]

        # Perform vector arithmetic
        target_v= (word2_v-word1_v)+word3_v

        # Create a dictionary mapping each word in the vocabulary to its corresponding vector
        # Exclude the input words from the possible results

        words_v= {}
        excluded_words= self.model.words.copy()
        excluded_words.remove(word1)
        excluded_words.remove(word2)
        excluded_words.remove(word3)
        
        for word in excluded_words:
            words_v[word]= self.model[word]
        
        # Find the word whose vector is closest to the result vector
        res_dist= float('inf')
        res_word= None
        for word, vec in words_v.items():
            dist= distance.cosine(vec, target_v)
            if dist<res_dist:
                res_dist= dist
                res_word= word

        return res_word
    
    
    def save_model(self, path):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)

    def load_model(self, path):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model= fasttext.load_model(path)

    def prepare(self, dataset, mode, save, path):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(path)
        if save:
            self.save_model(path)


if __name__ == "__main__":
    ft_model = FastText(method='skipgram')

    path= 'C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/indexer/index/'
    ft_data_loader = FastTextDataLoader(path)

    a= ft_data_loader.create_train_data()
    
    np.save('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/word_embedding/embedding_files/y_clustering.npy', np.array(a[1]))
    np.save('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/word_embedding/embedding_files/x_clustering.npy', np.array(a[0]))


    # X = ft_data_loader.create_train_data()[0]

    # # ft_model.train(X)
    # ft_model.prepare(None, "load", False, 'C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/word_embedding/model/FastText_model.bin')

    # print(10 * "*" + "Similarity" + 10 * "*")
    # word = 'queen'
    # neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    # for neighbor in neighbors:
    #     print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    # print(10 * "*" + "Analogy" + 10 * "*")
    # word1 = "king"
    # word2 = "man"
    # word3 = "queen"
    # print(f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
