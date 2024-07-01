import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import EvalPrediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from huggingface_hub import login, create_repo
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.top_n_genres)
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.y_test= None

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open(self.file_path, 'r') as f:
            self.dataset = json.load(f)

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres.
        """

        genres_count={}
        for entry in self.dataset:
            if entry['genres']:
                for genre in entry['genres']:
                    if genre in genres_count:
                        genres_count[genre]+= 1
                    else:
                        genres_count[genre]= 1

        sorted_genres= dict(sorted(genres_count.items(), key=lambda item: item[1], reverse=True))
        top_genres= {k: sorted_genres[k] for k in list(sorted_genres)[:self.top_n_genres]}

        filter_top_genre_movies=[]
        for entry in self.dataset:
            if entry['genres']:
                entry_top_genres= [genre for genre in entry['genres'] if genre in top_genres]
                if entry_top_genres:
                    entry['genres']= entry_top_genres
                    filter_top_genre_movies.append(entry)
        self.dataset= pd.DataFrame(filter_top_genre_movies)

        plt.figure(figsize=(10, 5))
        plt.bar(top_genres.keys(), top_genres.values(), color= plt.cm.viridis(np.linspace(0, 1, len(top_genres.keys()))))
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.title('Top Genres Distribution')
        plt.show()

    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """


        x= self.dataset["first_page_summary"].tolist()
        y= LabelEncoder().fit_transform(self.dataset['genres'].str[0].tolist())

        x_train, x_test_val, y_train, y_test_val = train_test_split(x, y, test_size= test_size, random_state=42)
        x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val,test_size= val_size, random_state=42)

        self.train_dataset= self.create_dataset(x_train, y_train)
        self.val_dataset= self.create_dataset(x_val, y_val)
        self.test_dataset= self.create_dataset(x_test, y_test)
        self.y_test= y_test


    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        encodings = self.tokenizer(list(map(str, encodings)), truncation=True, padding=True)
        return IMDbDataset(encodings, labels)

    def encode_labels(self, labels):
        """
        Encode the labels to numerical format.
        """
        return LabelEncoder().fit_transform(labels.str[0].tolist())

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """

        training_args = TrainingArguments(
            output_dir= './results',
            num_train_epochs= epochs,
            per_device_train_batch_size= batch_size,
            per_device_eval_batch_size= batch_size,
            warmup_steps= warmup_steps,
            weight_decay= weight_decay,
            logging_dir= './logs',
            logging_steps= 10,
            evaluation_strategy= "epoch",
            save_strategy= "epoch"
        )

        trainer = Trainer(
            model= self.model,
            args= training_args,
            train_dataset= self.train_dataset,
            eval_dataset= self.val_dataset,
            compute_metrics= self.compute_metrics
        )

        trainer.train()
        self.model= trainer

    def calculate_eval_scores(self, preds, labels):

        # p= np.mean(np.array([set(pred).issubset(set(label)) for pred, label in zip(preds, labels)]))
        # r= np.mean(np.array([set(label).issubset(set(pred)) for pred, label in zip(preds, labels)]))
        # f1=2*(p*r)/(p+r)
        # acc= np.mean(np.array([set(pred)==set(label) for pred, label in zip(preds, labels)]))


        p= precision_score(labels, preds, average='weighted')
        r= recall_score(labels, preds, average='weighted')
        f1= f1_score(labels, preds, average='weighted')
        acc= accuracy_score(labels, preds)

        return p, r, f1, acc

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """

        labels= pred.label_ids
        preds= pred.predictions.argmax(-1)

        p, r, f1, acc= self.calculate_eval_scores(preds, labels)

        return {
            'Precision': p,
            'Recall': r,
            'Accuracy': acc,
            'F1': f1
        }

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        pred, hid_s, att_w= self.model.predict(self.test_dataset)
        preds= np.argmax(pred, axis=1)

        p, r, f1, acc= self.calculate_eval_scores(preds, self.y_test)

        print(f' Precision: {p}\n Recall: {r}\n F1 Score: {f1}\n Accuracy: {acc}')


    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        self.model.save_model(model_name)
        self.tokenizer.save_pretrained(model_name)

        token= "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"
        login(token)
        repository_url= create_repo(repo_id=model_name)
        self.model.push_to_hub(model_name, token)
        self.tokenizer.push_to_hub(model_name,token)


class IMDbDataset(Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.encodings= encodings
        self.labels= labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        item= {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels']= torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)
