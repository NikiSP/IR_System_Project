import re
import nltk
import string
import json
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        nltk.download('punkt')
        nltk.download('wordnet')
        with open("Logic/core/stopwords.txt", 'r') as file:
            self.stopwords= [line.strip() for line in file]
        
        # nltk.download('stopwords')
        
        # self.stopwords= set(stopwords.words('english'))
        self.stemmer= PorterStemmer()
        self.lemmatizer= WordNetLemmatizer()
        
        self.documents= documents


    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
         # TODO
        preprocessed_docs= []
        for doc in self.documents:
            for field in doc:
                if isinstance(doc[field], str):
                    doc[field]= self.normalize(doc[field])
                elif isinstance(doc[field], list):
                    doc[field]= self.step_list_preprocess(doc[field])
            preprocessed_docs.append(doc)
            
        return preprocessed_docs

    def step_list_preprocess(self, list_field): 
        if not list_field: 
            return None
        
        new_list_field= []

        if isinstance(list_field[0], str):
            for a in list_field:
                new_list_field.append(self.normalize(a))
            return new_list_field
        
        for a in list_field:
            new_list_field.append(self.step_list_preprocess(a))
        return new_list_field

        
    def normalize(self, text: str):
        
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """  

        text= text.lower()  
        text= self.remove_links(text)
        text= self.remove_punctuations(text)
        tokens= self.remove_stopwords(text)
        
        normalized_tokens= [self.lemmatizer.lemmatize(tok) for tok in [self.stemmer.stem(token) for token in tokens]]

        return (' '.join(normalized_tokens))
    

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns= [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        
        for pattern in patterns:
            text= re.sub(pattern, '', text)
            
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        return word_tokenize(text)
        

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        tokens= self.tokenize(text)
        tokens= [tok for tok in tokens if tokens not in self.stopwords]
        return tokens

# with open('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/IMDB_crawled.json', 'r') as f:
#     documents= json.load(f)

# preprocessor= Preprocessor(documents)
# processed_docs= preprocessor.preprocess()

# with open('preprocessed_documents.json', 'w') as f:
#     json.dump(list(processed_docs), f)

with open('preprocessed_documents.json', 'r') as f:
    documents= json.load(f)
print(len(documents))