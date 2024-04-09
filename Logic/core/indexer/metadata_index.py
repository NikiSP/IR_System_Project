from index_reader import Index_reader
from indexes_enum import Indexes, Index_types
import json
import document_lengths_index
from statistics import mean 

class Metadata_index:
    def __init__(self, path='C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/indexer/index/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """
        
        #TODO
        self.path= path
        self.documents= self.read_documents()
        self.indexes = {
            Indexes.STARS.value: Index_reader(self.path, index_name=Indexes.STARS, index_type=Index_types.DOCUMENT_LENGTH).index,
            Indexes.GENRES.value: Index_reader(self.path, index_name=Indexes.GENRES, index_type=Index_types.DOCUMENT_LENGTH).index,
            Indexes.SUMMARIES.value: Index_reader(self.path, index_name=Indexes.SUMMARIES, index_type=Index_types.DOCUMENT_LENGTH).index,
        }
        
        self.metadata_index= self.create_metadata_index()
                
        self.store_metadata_index(path)

        
    def read_documents(self):
        """
        Reads the documents.
        
        """
        #TODO
        return Index_reader(self.path, index_name=Indexes.DOCUMENTS, index_type=None).index
    
    def create_metadata_index(self):    
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['averge_document_length'] = {
            'stars': self.get_average_document_field_length('stars'),
            'genres': self.get_average_document_field_length('genres'),
            'summaries': self.get_average_document_field_length('summaries')
        }
        metadata_index['document_count'] = len(self.documents)
        

        return metadata_index
    
    def get_average_document_field_length(self,where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """
        
        
        #TODO
        # len_sum= []
        # for length in self.indexes[where].values():
        #     len_sum.append(int(length))
        len_sum= 0
        
        for doc in self.documents.values():
            if doc[where]:
                for section in doc[where]:
                    if where=='stars':
                        print(len(doc[where]))
                        print(section)
                        print(len(section))
                    len_sum+= (len(section.split()))
                
        
        return (len_sum/len(self.documents))
        
    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        path =  path + Indexes.DOCUMENTS.value + '_' + Index_types.METADATA.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.metadata_index, file, indent=4)


    
if __name__ == "__main__":
    meta_index = Metadata_index()