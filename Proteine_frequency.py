import numpy as np
from matplotlib import pyplot as plt
import random
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import sklearn.preprocessing as pp
import seaborn as sns

random.seed(420)
# import dataset

plt.style.use('dark_background')

dataset_filename = '00_datasets/Life_Science/Codon_Usage/codon_usage.csv'
df = pd.read_csv(dataset_filename, low_memory=False)

df.describe()

df.info()

df.Kingdom.unique()

print(len(df.SpeciesName.unique()))
print(len(df.SpeciesID.unique()))

#setting string values of UUU and UCC to NaN
df['UUU'] = pd.to_numeric(df['UUU'], errors='coerce')
df['UUC'] = pd.to_numeric(df['UUC'], errors='coerce')
df = df.dropna()


###########################################################
#
#                               EDA
#
###########################################################
#kingdom_frequency = df.drop(columns=['DNAtype', 'SpeciesID', 'SpeciesName'])  --cagata
#sns.pairplot(kingdom_frequency, hue='Kingdom')
###########################################################
X = df.iloc[:, 5:69]
df2 = df.drop(['DNAtype', 'SpeciesID','SpeciesName'],axis = 1)

sns.scatterplot(data=df2, x=df2.drop(['Kingdom','Ncodons'], axis=1).index)
aminoacids = df2.T.drop(['Kingdom','Ncodons']).index
aminoacids = df2.index

#########################################################################
# For each kingodom calculate the maean values of each codon
#########################################################################



kingdoms_codons = pd.DataFrame(columns=df.Kingdom.unique(), index=X.keys())

#counting saples for each kingdom :

plt.figure(1, figsize=(10, 10))
sns.countplot(y='Kingdom', data=df)
plt.show()

df.set_index('Kingdom')
for kingdom in kingdoms_codons.columns:
    for codon in kingdoms_codons.index:
        kingdoms_codons[kingdom].loc[codon] = np.mean(df[codon].loc[df.Kingdom == kingdom])

kc_T = kingdoms_codons.T
for codon in kc_T:
        kc_T[codon] = kc_T[codon].astype('float64')


plt.figure(1, figsize=(21, 7))
sns.heatmap(kc_T)
plt.show()

plt.figure(1, figsize=(32,11))
sns.lineplot(data=kc_T.T, linewidth=1.0)
plt.show()
Y = df.Kingdom
###########################################################
#
#                Classifiers Models
#
############################################################
NN = MLPClassifier(hidden_layer_sizes=(1000,),
                   max_iter=1000,
                   early_stopping=True,
                   validation_fraction=0.2,
                   activation='relu',
                   solver='adam',
                   verbose=True)
param_grid = { 'hidden_layer_sizes' : [(100, 100), (500,500)], }

