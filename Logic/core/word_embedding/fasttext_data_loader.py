import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import sys
import os
import numpy as np

current= os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(current))))

from core.indexer.index_reader import Index_reader
from core.indexer.indexes_enum import Indexes

class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path
        self.docs= Index_reader(self.file_path, index_name=Indexes.DOCUMENTS, index_type= None).index
        
        
    def extract_section_data(self, section_name):
        res= []
        count= 0
        for movie in (self.docs.values()):
            if movie[section_name]:
                res.append(movie[section_name])
            else:
                res.append('')
            count+=1
            if count>=1000:
                return res
        return res
                

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        df= pd.DataFrame({
            'synopsis': self.extract_section_data('synopsis'),
            'summaries': self.extract_section_data('summaries'),
            'reviews': self.extract_section_data('reviews'),
            'title': self.extract_section_data('title'),
            'genres': self.extract_section_data('genres')
        })
        return df


    def extract_string_section(self, df, name):
        for i, list_of_strings in enumerate(df[name]):
            res= ''
            for text in list_of_strings:
                res+= (text[0]+' ')
            df.at[i, name]= res.strip()
        
        return df

    
    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).
        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        
        df= self.read_data_to_df()
        # the docs we used in order to create DOCUMENTS index were already preprocessed, so we don't need to repeat the same thing here
        # just need to turn them into one whole string 
        
        df= self.extract_string_section(df, 'reviews')
        
        encoder= LabelEncoder()
        
        df['genres']= df['genres'].apply(lambda x: x[0] if isinstance(x, list) else x)

        df['genres']= encoder.fit_transform(df['genres'].astype(str))   
        return (df['reviews'].values.astype(str), np.array(df['genres']))
    
        


