import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# from .basic_classifier import BasicClassifier
# from .data_loader import ReviewLoader

from basic_classifier import BasicClassifier
from data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc

        Returns
        -------
        self
            Returns self as a classifier
        """
        self.classes= np.unique(y)
        self.num_classes= len(self.classes)
        self.number_of_samples, self.number_of_features = x.shape

        self.prior= np.zeros(self.num_classes)
        self.probs= np.zeros((self.num_classes, self.number_of_features))

        for cl in self.classes:
            x_cl= x[y==cl]
            self.prior[cl]= x_cl.shape[0]/self.number_of_samples
            self.probs[cl, :]= (x_cl.sum(axis=0)+self.alpha)/(x_cl.sum()+(self.alpha*self.number_of_features))

        self.log_probs= np.log(self.probs)
        return self

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        res= x@self.log_probs.T+np.log(self.prior)
        return np.argmax(res, axis=1)
        

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        return classification_report(y, self.predict(x))

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        pos_preds= np.sum(self.predict(self.cv.transform(sentences).toarray())==1)
        return (pos_preds/len(preds))*100


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    loader= ReviewLoader('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/classification/IMDB_Dataset.csv')
    loader.load_data(True, 'C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/word_embedding/model/FastText_model.bin')
    reviews, sentiments= loader.get_data()
    
    count_vectorizer= CountVectorizer(max_features= 3000)
    embeds= count_vectorizer.fit_transform(reviews).toarray()
    labels= LabelEncoder().fit_transform(sentiments)
   
    x_train, x_test, y_train, y_test = train_test_split(embeds, labels, test_size=0.2, random_state=42)

    naivebayes_classifier= NaiveBayes(count_vectorizer=count_vectorizer, alpha=1)
    naivebayes_classifier.fit(x_train, y_train)

    print(naivebayes_classifier.prediction_report(x_test, y_test))

