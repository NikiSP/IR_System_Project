import numpy as np
import pandas as pd
import tqdm
import sys 
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

sys.path.append("../")
current= os.path.dirname(__file__)
current= os.path.join(current, '..')
sys.path.append(os.path.dirname((os.path.abspath(current))))


# from ..word_embedding.fasttext_model import FastText
# from ..utility.preprocess import Preprocessor

from core.word_embedding.fasttext_model import FastText
from core.utility.preprocess import Preprocessor

class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = None
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self, load, model_path):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        df= pd.read_csv(self.file_path)
        
        reviews= df['review'].astype(str).tolist()
     
        if load:
            with open('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/classification/preprocessed_reviews.json', 'r') as f:
                self.review_tokens= json.load(f)
        else:
            preprocessor= Preprocessor(reviews)
            self.review_tokens=  preprocessor.preprocess()
            with open('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/classification/preprocessed_reviews.json', 'w') as f:
                json.dump(list(self.review_tokens), f)
                
        
        
        # preprocessed_tokens= preprocessor.preprocess()
        # for string in preprocessed_tokens:
        #     self.review_tokens.append(string.split())
        
        self.sentiments= df['sentiment'].astype(str).tolist()
        
        self.fasttext_model= FastText()
        self.fasttext_model.prepare(None, "load", False, path)
        


    def get_data(self):
        return self.review_tokens, self.sentiments
    
    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        for toks in self.review_tokens:
            self.embeddings.append(self.fasttext_model.get_query_embedding(toks))
            
    def get_embedding_values(self):
        return self.embeddings

    def split_data(self, test_data_ratio):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        encoder= LabelEncoder()
        sentiment_labels= encoder.fit_transform(self.sentiments) 
        x_train, x_test, y_train, y_test = train_test_split(self.embeddings, sentiment_labels, test_size=test_data_ratio, random_state=42)
        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

