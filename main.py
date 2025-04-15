import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from randomForest import randomForest as rf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import text_preprocessing
import load_datasets

df_profiles = load_datasets.load_profiles_dataset()
df_comments = load_datasets.load_comments_dataset()

