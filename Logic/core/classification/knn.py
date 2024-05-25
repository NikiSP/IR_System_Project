import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

# from .basic_classifier import BasicClassifier
# from .data_loader import ReviewLoader
from basic_classifier import BasicClassifier
from data_loader import ReviewLoader

class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors
        self.x_train= None
        self.y_train= None
        
    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

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
        self.x_train= x
        self.y_train= y
        return 

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
        preds= []
        for point in x:
            
            dists= np.linalg.norm(self.x_train-point, axis=1)
            labels= self.y_train[np.argsort(dists)[:self.k]]
            
            labels, counts= np.unique(labels, return_counts=True)
            preds.append(labels[counts.argmax()])
            
        return np.array(preds)

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


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """

    loader= ReviewLoader('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/classification/IMDB_Dataset.csv')
    loader.load_data(True, 'C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/word_embedding/model/FastText_model.bin')
    loader.get_embeddings()
    x_train, x_test, y_train, y_test= loader.split_data(test_data_ratio=0.2)
    
    knn_classifier= KnnClassifier(3)
    knn_classifier.fit(x_train, y_train)

    print(knn_classifier.prediction_report(x_test, y_test))
