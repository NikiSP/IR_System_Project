import numpy as np
from typing import List
from statistics import mean 
import wandb


class Evaluation:

    def __init__(self, name: str):
            self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        precision = 0.0

        # TODO: Calculate precision here
        for retrieveds, relevants in zip(predicted, actual):
            for retrieved in retrieveds:
                val= 0
                if retrieved in relevants:
                    val+= 1
            precision+= val/len(retrieveds)

        return (precision/len(actual))
    
    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        recall = 0.0

        # TODO: Calculate recall here
        for retrieveds, relevants in zip(predicted, actual):
            for retrieved in retrieveds:
                val= 0
                if retrieved in relevants:
                    val+= 1
            recall+= val/len(relevants)

        return (recall/len(actual))
    
    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results    
        """
        f1 = 0.0
        precision= self.calculate_precision(actual, predicted)
        recall= self.calculate_recall(actual, predicted)
        
        f1= (2*precision*recall)/(precision+recall)
        # TODO: Calculate F1 here

        return f1
    
    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> list:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Average Precision of the predicted results
        """
        AP = 0.0
        
        AP_values= []
        total_count= 0
        relevant_count= 0
        # TODO: Calculate AP here
        
        for i in range(len(predicted)):
            
            precisions= []
            for doc in predicted[i]:
                total_count+= 1
                if doc in actual[i]:
                    relevant_count+= 1
                    precisions.append(relevant_count/total_count)
            AP_values.append(mean(precisions))
            
        return list(AP_values)
    
    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        # TODO: Calculate MAP here
        
        return mean(self.calculate_AP(actual, predicted))

    
    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> list:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The DCG of the predicted results
        """
        NDCGs= []
        if not predicted or not predicted[0]:
            return list()
        
        rel= np.ones((len(predicted), len(predicted[0]))) #?
        perfect_ranking_rel= np.ones((len(predicted), len(predicted[0])))
        
        for i in range(len(predicted)):
            DCGs= []
            IDCGs= []
            for j in range(len(predicted[i])):
                DCGs.append(((2**(rel[i][j]))-1)/np.log2(j+2))
                IDCGs.append(((2**(perfect_ranking_rel[i][j]))-1)/np.log2(j+2))
                
            NDCGs.append(mean(DCGs)/mean(IDCGs))
            
        # TODO: Calculate DCG here

        return list(NDCGs)
    
    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        # TODO: Calculate NDCG here
        
        return mean(self.cacluate_DCG(actual, predicted))

    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> list:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Reciprocal Rank of the predicted results
        """
        RRs= []

        # TODO: Calculate MRR here
        for i in range(len(predicted)):
            for j in range(len(predicted[i])):
                if predicted[i][j] in actual[i][j]:
                    RRs.append(1/(j+1))
                    break
        
        return list(RRs)
    
    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        
        # TODO: Calculate MRR here
        return mean(self.cacluate_RR(actual, predicted))

    

    def print_evaluation(self, precision, recall, f1, ap, map_val, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        print(f"name = {self.name}")

        #TODO: Print the evaluation metrics
        print("precision:", precision)
        print("recall:", recall)
        print("f1:", f1)
        print("ap:", ap)
        print("map:", map_val)
        print("dcg:", dcg)
        print("ndcg:", ndcg)
        print("rr:", rr)
        print("mrr:", mrr)

    def log_evaluation(self, precision, recall, f1, ap, map_v, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics
      
        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results
            
        """
        
        #TODO: Log the evaluation metrics using Wandb
        wandb.init('IR_System_Project')
        wandb.log({
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Average Precision": ap,
            "Mean Average Precision": map_v,
            "Discounted Cumulative Gain": dcg,
            "Normalized Discounted Cumulative Gain": ndcg,
            "Reciprocal Rank": rr,
            "Mean Reciprocal Rank": mrr
        })


        
    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results
            
        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        #call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)



