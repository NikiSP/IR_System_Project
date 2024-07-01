      
import json
from .indexes_enum import Indexes, Index_types


class Tiered_by_year_index:
    def __init__(self, path="C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/indexer/index/"):
        """
        Initializes the Tiered_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """
        with open('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/IMDB_crawled.json', 'r') as f:
            self.documents= json.load(f)


        self.index = {
            Indexes.STARS: Index_reader(path, index_name=Indexes.STARS, index_type=None).index,
            Indexes.GENRES: Index_reader(path, index_name=Indexes.GENRES,index_type=None).index,
            Indexes.SUMMARIES: Index_reader(path, index_name=Indexes.SUMMARIES,index_type=None).index,
            Indexes.YEAR: Index_reader(path, index_name=Indexes.YEAR,index_type=None).index,
        }
        # feel free to change the thresholds
        self.tiered_index = {
            Indexes.STARS: self.convert_to_tiered_index(3, 2, Indexes.STARS),
            Indexes.SUMMARIES: self.convert_to_tiered_index(10, 5, Indexes.SUMMARIES),
            Indexes.GENRES: self.convert_to_tiered_index(1, 0, Indexes.GENRES)
        }
        self.store_tiered_by_year_index(path, Indexes.STARS)
        self.store_tiered_by_year_index(path, Indexes.SUMMARIES)
        self.store_tiered_by_year_index(path, Indexes.GENRES)

    def convert_to_tiered_index(
        self, first_tier_threshold: int, second_tier_threshold: int, index_name
    ):
        """
        Convert the current index to a tiered index.

        Parameters
        ----------
        first_tier_threshold : int
            The threshold for the first tier
        second_tier_threshold : int
            The threshold for the second tier
        index_name : Indexes
            The name of the index to read.

        Returns
        -------
        dict
            The tiered index with structure of 
            {
                "first_tier": dict,
                "second_tier": dict,
                "third_tier": dict
            }
        """
        if index_name not in self.index:
            raise ValueError("Invalid index type")

        current_index = self.index[index_name]
        tiers= [{} for i in range(13)]
        
        for term, docs in current_index.items():
            temp_tires= [{} for i in range(13)]
            
            for movie, tf in docs.items():
                if not self.index[Indexes.YEAR][movie] or not self.index[Indexes.YEAR][movie].isnumeric():
                    continue
                # print(self.index[Indexes.YEAR][movie])
                year= int(self.index[Indexes.YEAR][movie][0:4])
                cent= int(year/1000)
                dec= int((year%100)/10)
                indx= ((cent-1)*10) + dec    
 
                temp_tires[indx][movie]= tf 
                
            for i in range(13):
                if temp_tires[i]:
                    tiers[i][term]= temp_tires[i]


        return tiers

    def store_tiered_by_year_index(self, path, index_name):
        """
        Stores the tiered index to a file.
        """
        path = path + index_name.value + "_" + Index_types.YEAR.value + "_index.json"
        with open(path, "w") as file:
            json.dump(self.tiered_index[index_name], file, indent=4)



if __name__ == "__main__":
    tiered = Tiered_by_year_index(
        path="C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/indexer/index/"
    )
