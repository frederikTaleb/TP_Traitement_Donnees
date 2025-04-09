import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class Predicteur:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.predictor = None

    def train(self,X, y):
        """
        Entraîne un random forest
        X : les exemples 
        y : les étiquettes associées
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        cross_validation_results = cross_validate(self.model, X_train, y_train, cv=5, return_estimator=True)

        esitmator_index = np.argmax(cross_validation_results['test_score'])
        print(cross_validation_results['test_score'])
        print(esitmator_index)

        self.predictor = cross_validation_results['estimator'][esitmator_index]
    
        test_pred = self.predictor.predict(X_test)

        test_accuracy = accuracy_score(y_test, test_pred)

        print(test_accuracy)
    
    def predict(self, X):
        return self.predictor.predict(X)

        