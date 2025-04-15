import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class randomForest():
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self.model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
    
    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.samples, self.labels, test_size=0.2, random_state=42, stratify=self.labels)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_pred, y_test)
        self.cm = confusion_matrix(y_pred,y_test)
    
    def predict(self, samples):
        return self.model.predict(samples)
    
    def getImportance(self):
        importances = self.model.feature_importances_
        feature_names = self.samples.columns
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        print(f"\nImportance des attributs pour prédire '{self.labels.name}':")
        print(feature_importance_df.head(15))

        # Plot feature importances
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15)) # Plot top N features
        plt.title(f'Importance des attributs pour prédire {self.labels.name}')
        plt.tight_layout()
        plt.show()