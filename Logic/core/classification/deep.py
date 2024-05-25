import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# from .data_loader import ReviewLoader
# from .basic_classifier import BasicClassifier

from data_loader import ReviewLoader
from basic_classifier import BasicClassifier

class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        super().__init__()
        self.test_loader = None
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.best_model = self.model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # self.device= 'cpu'
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else self.device
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def fit(self, x, y):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        x: np.ndarray
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        loader= DataLoader(ReviewDataSet(x, y), batch_size=self.batch_size, shuffle=True)

        best_f1= 0.0
        for epoch in range(self.num_epochs):
            self.model.train()
            loss= 0.0
            for emb, lbl in loader:
                emb, lbl= emb.to(self.device), lbl.to(self.device)
                self.optimizer.zero_grad()
                
                samp_loss= self.criterion(self.model(emb), lbl)
                samp_loss.backward()
                self.optimizer.step()
                loss+= samp_loss.item()
            
            if self.test_loader:
                eval_loss, predicted_labels, true_labels, f1_score_macro= self._eval_epoch(self.test_loader, self.model)
                if f1_score_macro>best_f1:
                    best_f1=f1_score_macro
                    self.best_model = self.model.state_dict()

        self.model.load_state_dict(self.best_model)
        return self
    
        return self

    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        self.model.eval()
        with torch.no_grad():
            x= torch.FloatTensor(x).to(self.device)
            predicted_labels= self.model(x).argmax(dim=1).cpu().numpy()
        return predicted_labels
    

    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader, model):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """
        model.eval()
        
        eval_loss= 0.0
        predicted_labels= []
        true_labels= []

        with torch.no_grad():
            for emb, lbl in dataloader:
                emb, lbl= emb.to(self.device), lbl.to(self.device)
                preds= model(emb)
                
                eval_loss+= self.criterion(preds, lbl).item()
                
                predicted_labels.extend(preds.argmax(dim=1).cpu().numpy())
                true_labels.extend(lbl.cpu().numpy())

        return (eval_loss/len(dataloader)), predicted_labels, true_labels, (f1_score(true_labels, predicted_labels, average='macro'))


    def set_test_dataloader(self, X_test, y_test):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        
        self.test_loader = DataLoader(ReviewDataSet(X_test, y_test), batch_size=self.batch_size, shuffle=False)
        return self

    def prediction_report(self, x, y):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """

        return classification_report(y, self.predict(x))

# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    
    loader= ReviewLoader('C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/classification/IMDB_Dataset.csv')
    loader.load_data(True, 'C:/Users/FasleJadid/Desktop/IRProject/IR_System_Project/Logic/core/word_embedding/model/FastText_model.bin')
    loader.get_embeddings()
    x_train, x_test, y_train, y_test= loader.split_data(test_data_ratio=0.2)

    deep_classifier= DeepModelClassifier(in_features= x_train.shape[1], num_classes=len(np.unique(y_train)), batch_size= 64, num_epochs= 3)
    deep_classifier.set_test_dataloader(x_test, y_test)
    deep_classifier.fit(x_train, y_train)

    print(deep_classifier.prediction_report(x_test, y_test))