import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import load_datasets
import feature_eng
import Predicteur_random_forest as rf


df_profiles = load_datasets.load_profiles_mbti_ocean_all_scores()

X_train = feature_eng.discretize(df_profiles, ['agreeableness', 'openness','conscientiousness','extraversion','neuroticism'],10)
y_train = {'introverted': df_profiles['introverted'],'intuitive':df_profiles['intuitive'],'thinking':df_profiles['thinking'],'perceiving':df_profiles['perceiving']}

model_introverted = rf.Predicteur()
model_intuitive = rf.Predicteur()
model_thinking = rf.Predicteur()
model_perceiving = rf.Predicteur()

model_introverted.train(X_train, y_train['introverted'])
model_intuitive.train(X_train, y_train['intuitive'])
model_thinking.train(X_train, y_train['thinking'])
model_perceiving.train(X_train, y_train['perceiving'])

df_profiles_ocean = load_datasets.load_profiles_ocean_scores()
df_profiles_ocean_no_mbti = df_profiles_ocean.isna(subset=['introverted', 'intuitive','thinking','perceiving'])
X_predict = feature_eng.discretize(df_profiles_ocean_no_mbti,['agreeableness', 'openness','conscientiousness','extraversion','neuroticism'],10)

predict_introverted = model_introverted.predict(X_predict)
predict_intuitive = model_intuitive.predict(X_predict)
predict_thinking= model_thinking.predict(X_predict)
predict_perceiving = model_perceiving.predict(X_predict)

df_profiles_ocean_no_mbti.loc[:, 'introverted'] = np.array(predict_introverted)
df_profiles_ocean_no_mbti.loc[:, 'intuitive'] = np.array(predict_intuitive)
df_profiles_ocean_no_mbti.loc[:, 'thinking'] = np.array(predict_thinking)
df_profiles_ocean_no_mbti.loc[:, 'perceiving'] = np.array(predict_perceiving)

df_profiles_ocean_no_mbti[['author','introverted', 'intuitive','thinking','perceiving']].to_csv('datasets/author_mbti_predict.csv', index=False)