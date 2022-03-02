import numpy as np
from matplotlib import pyplot as plt
import random
import time
import plotly.express as px
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import silhouette_samples, \
    silhouette_score, \
    f1_score, precision_score, accuracy_score, auc, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.preprocessing as pp
import matplotlib.cm as cm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


###########################################################################
#                               USEFUL FUNCTIONS
############################################################################


# TODO: function for controlling classification
def first_kingdom_classifier(data):
    dic = {'vrl': 'vrl', 'phg': 'vrl',
           'bct': 'prk', 'arc': 'prk',
           'pln': 'euk', 'mam': 'euk',
           'inv': 'euk', 'vrt': 'euk',
           'rod': 'euk', 'pri': 'euk',
           }
    return data.replace({'Kingdom': dic}).Kingdom


def second_kingdom_classifier(dataf):
    dic = {'mam': 'ani',
           'vrt': 'ani',
           'rod': 'ani',
           'inv': 'ani',
           'pri': 'ani'}
    return dataf.replace({'Kingdom': dic}).Kingdom


def test_kingdom_classifier(dataf):
    dic = {'mam': 'ani_vrt',
           'vrt': 'ani_vrt',
           'rod': 'ani_vrt',
           'inv': 'ani_inv',
           'pri': 'ani_vrt'}
    return dataf.replace({'Kingdom': dic}).Kingdom


# ------------------- outliners remover ----------------------------------- #
def outlier_remover(dataframe, quantile_sup=0.99, quantile_inf=0.0, features=None):
    X_outlined = pd.DataFrame(columns=np.array(dataframe.columns))
    if features is None:
        for feature in dataframe.columns:
            X_outlined = dataframe[
                dataframe[feature].between(dataframe[feature].quantile(quantile_inf),
                                           dataframe[feature].quantile(quantile_sup))]
    else:
        for feature in dataframe.columns:
            X_outlined = dataframe[
                dataframe[feature].between(dataframe[feature].quantile(quantile_inf),
                                           dataframe[feature].quantile(quantile_sup))]

    return X_outlined


# ------------------- Silhouette Plotting ----------------------------------#


def silhouette_plot(labels, data):
    cluster = len(np.unique(labels))
    fig_s = plt.figure()
    ax = fig_s.add_subplot(111)
    # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters
    ax.set_ylim([0, len(data) + (cluster + 1) * 10])
    # Compute Silhouette
    silhouette_avg = silhouette_score(data, labels, metric='euclidean')
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, labels)
    y_lower = 10
    for clust in range(cluster):
        print(clust)
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == clust]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.gist_rainbow(float(clust) / cluster)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color,
                         alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(clust))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax.set_title("Silhouette plot for the various clusters.")
    ax.set_xlabel("Silhouette coefficient value")
    ax.set_ylabel("Cluster")
    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    fig_s.show()


# -------------------- ROC Plotting --------------------------------
def ROC_plotting(y_test, y_pred, title):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for query_class in np.unique(y_test):
        fpr[query_class], tpr[query_class], threshold = roc_curve(y_test == query_class, y_pred == query_class)
        roc_auc[query_class] = auc(fpr[query_class], tpr[query_class])

    plt.figure(constrained_layout=True)
    for query_class in np.unique(y_test):
        plt.plot(fpr[query_class], tpr[query_class], label='ROC classe {0} (area ={1:0.2f})'
                                             ''.format(query_class, roc_auc[query_class]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


# ---------------- 2D Visualizator ----------------------------
def visualizator_2d(data, labels=None, title=None, indexes=None, palette=None):
    visualizator2d = PCA(n_components=2, random_state=420)
    transformed_X = visualizator2d.fit_transform(data)
    print("total explained variance:", visualizator2d.explained_variance_ratio_.sum())
    if labels is None:
        plt.figure(figsize=(20, 10), constrained_layout=True)
        plt.subplot(1, 2, 1)
        if indexes is None:
            sns.scatterplot(x=transformed_X[:, 0], y=transformed_X[:, 1], palette=palette, hue=data_kingdom['Kingdom'])
            plt.subplot(1, 2, 2)
            sns.scatterplot(x=transformed_X[:, 0], y=transformed_X[:, 1], palette=palette, hue=data_kingdom['DNAtype'])
        else:
            sns.scatterplot(x=transformed_X[:, 0], y=transformed_X[:, 1], palette=palette,
                            hue=data_kingdom['Kingdom'].loc[indexes])
            plt.subplot(1, 2, 2)
            sns.scatterplot(x=transformed_X[:, 0], y=transformed_X[:, 1], palette=palette,
                            hue=data_kingdom['DNAtype'].loc[indexes])
    else:
        plt.figure(figsize=(10, 10), constrained_layout=True)
        sns.scatterplot(transformed_X[:, 0], transformed_X[:, 1], palette=cm.gist_rainbow, hue=labels)
    if title is not None:
        plt.suptitle(title)
    plt.show()

    def visualizator_2d(data, labels=None, title=None, indexes=None, palette=None):
        visualizator2d = PCA(n_components=2, random_state=420)
        transformed_X = visualizator2d.fit_transform(data)
        print("total explained variance:", visualizator2d.explained_variance_ratio_.sum())
        if labels is None:
            plt.figure(figsize=(20, 10), constrained_layout=True)
            plt.subplot(1, 2, 1)
            if indexes is None:
                sns.scatterplot(x=transformed_X[:, 0], y=transformed_X[:, 1], palette=palette,
                                hue=data_kingdom['Kingdom'])
                plt.subplot(1, 2, 2)
                sns.scatterplot(x=transformed_X[:, 0], y=transformed_X[:, 1], palette=palette,
                                hue=data_kingdom['DNAtype'])
            else:
                sns.scatterplot(x=transformed_X[:, 0], y=transformed_X[:, 1], palette=palette,
                                hue=data_kingdom['Kingdom'].loc[indexes])
                plt.subplot(1, 2, 2)
                sns.scatterplot(x=transformed_X[:, 0], y=transformed_X[:, 1], palette=palette,
                                hue=data_kingdom['DNAtype'].loc[indexes])
        else:
            plt.figure(figsize=(10, 10), constrained_layout=True)
            sns.scatterplot(transformed_X[:, 0], transformed_X[:, 1], palette=cm.gist_rainbow, hue=labels)
        if title is not None:
            plt.suptitle(title)
        plt.show()


def visualizator_2d_ref():
    visualizator2d = PCA(n_components=2, random_state=420)
    transformed_X = visualizator2d.fit_transform(X_nucle)
    transformed_XM = visualizator2d.fit_transform(X_MMS)
    print("total explained variance:", visualizator2d.explained_variance_ratio_.sum())

    plt.figure(figsize=(20, 20), constrained_layout=True)
    plt.subplot(2, 2, 1)
    sns.scatterplot(x=transformed_X[:, 0], y=transformed_X[:, 1], hue=data_kingdom['Kingdom'])
    plt.subplot(2, 2, 2)
    sns.scatterplot(x=transformed_X[:, 0], y=transformed_X[:, 1], hue=data_kingdom['DNAtype'])
    plt.subplot(2, 2, 3)
    sns.scatterplot(transformed_XM[:, 0], transformed_XM[:, 1], hue=data_kingdom['Kingdom'])
    plt.subplot(2, 2, 4)
    sns.scatterplot(transformed_XM[:, 0], transformed_XM[:, 1], hue=data_kingdom['DNAtype'])
    plt.suptitle("Reference labels")
    plt.show()


def visualizator_2d_clustering(labels=None, title=None, palette=None):
    visualizator2d = PCA(n_components=2, random_state=420)
    transformed_XN = visualizator2d.fit_transform(X_nucle)
    transformed_XM = visualizator2d.fit_transform(X_MMS)
    transformed_X4 = visualizator2d.fit_transform(X_4d)
    transformed_X15 = visualizator2d.fit_transform(X_15d)
    plt.figure(figsize=(20, 20), constrained_layout=True)
    plt.subplot(2, 2, 1)
    sns.scatterplot(x=transformed_XN[:, 0], y=transformed_XN[:, 1], palette=palette, hue=labels[0])
    plt.title("X_nucle")
    plt.subplot(2, 2, 2)
    sns.scatterplot(x=transformed_XM[:, 0], y=transformed_XM[:, 1], palette=palette, hue=labels[1])
    plt.title("X_MMS")
    plt.subplot(2, 2, 3)
    sns.scatterplot(x=transformed_X4[:, 0], y=transformed_X4[:, 1], palette=palette, hue=labels[2])
    plt.title("X_4")
    plt.subplot(2, 2, 4)
    sns.scatterplot(x=transformed_X15[:, 0], y=transformed_X15[:, 1], palette=palette, hue=labels[3])
    plt.title("X_15")
    if title is not None:
        plt.suptitle(title)
    plt.show()


# ---------------- 3D Visualizator -------------------------------
def visualizator_3d(data, labels=None, indexes=None):
    pca_3D = PCA(n_components=3, random_state=420)
    transformed_3d = pca_3D.fit_transform(data)
    total_var = pca_3D.explained_variance_ratio_.sum() * 100
    if indexes is None:
        color = data_kingdom['Kingdom']
        symbol = data_kingdom['DNAtype']
    else:
        color = data_kingdom['Kingdom'].loc[indexes]
        symbol = data_kingdom['DNAtype'].loc[indexes]
    if labels is None:
        fig = px.scatter_3d(
            transformed_3d, x=0, y=1, z=2, color=color,
            symbol=symbol,
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )
    else:
        fig = px.scatter_3d(
            transformed_3d, x=0, y=1, z=2, color=labels,
            title=f'Total Explained Variance: {total_var:.2f}%',
            labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
        )

    fig.show()


# --------------- 3D visualizator (img) -----------------------------


random.seed(420)

plt.style.use('dark_background')

dataset_filename = 'Codon_Usage/codon_usage.csv'
df = pd.read_csv(dataset_filename, low_memory=False)

df.describe()
df.info()
df.Kingdom.unique()
# ----------------------- setting useful variables----------------------------#

# setting string values of UUU and UCC to NaN
df['UUU'] = pd.to_numeric(df['UUU'], errors='coerce')
df['UUC'] = pd.to_numeric(df['UUC'], errors='coerce')
df = df.dropna()  # Drop NaN values
meta_columns = ['Kingdom', 'DNAtype', 'SpeciesID', 'SpeciesName']

###############################################################################
#                               EDA
###############################################################################

"""
First count plot of kingdom class and Dna type class
"""
king_dnat = plt.figure(1, figsize=(20, 10), constrained_layout=True)
king = king_dnat.add_subplot(1, 2, 1)
king = sns.countplot(y='Kingdom', data=df)
dnat = king_dnat.add_subplot(1, 2, 2)
dnat = sns.countplot(y='DNAtype', data=df)
king.title.set_text('Kingdom class countplot')
dnat.title.set_text('DNAtype class countplot')
king_dnat.show()
king_dnat.savefig('img/proteine_frequency.png')
# --------------------- Get rid of PLM class and all DNAtype exept of 0-1-2
df = df[df['Kingdom'] != 'plm']
df = df.loc[df['DNAtype'].between(0, 2)]

"""
heatmap of kingdoms and
"""
kingdom_labels = test_kingdom_classifier(df)
data_kingdom = df.drop('Kingdom', axis=1)
data_kingdom = pd.concat([data_kingdom, kingdom_labels], axis=1)

heat_k = plt.figure(figsize=(18, 30), constrained_layout=True)
heat_k_x = heat_k.add_subplot(1, 1, 1)
heat_k_x = sns.heatmap(
    data_kingdom.drop(['SpeciesName', 'DNAtype', 'SpeciesID', 'Ncodons'],
                      axis=1).set_index('Kingdom').sort_index())
heat_k.show()
heat_k.savefig('img/heatmap_kingdom.png')

data_kingdom = data_kingdom.drop(['SpeciesID', 'SpeciesName'], axis=1)

bases = ['A', 'C', 'U', 'G']
aminoacids = data_kingdom.drop(['Kingdom', 'DNAtype', 'Ncodons'], axis=1).columns
for base in bases:
    n = 0
    plt.figure(1, figsize=(16, 16), constrained_layout=True)
    for codon in aminoacids:

        if codon.startswith(base):
            n += 1
            plt.subplot(4, 4, n)
            sns.boxplot(y="Kingdom", x=codon, data=data_kingdom)
            # sns.stripplot(x=codon, y="Kingdom",data=df2)
    plt.show()

"""
3 type of X : 

X_Ncodon = frequency analysis of codons (64 features) and 1 feature of the number of codons
-X_95 = value of X_Ncodons without outliers (0.95) => removed
X_nucle = frequency usage of codonos without ncodons

#X1 = df.drop(meta_columns, axis=1)

X1 was removed beacuse metacolumns included Ncodons 
removing the number of codons result in a average lower score
"""

X_Ncodon = data_kingdom.drop(['Kingdom', 'DNAtype'], axis=1)
X_nucle = X_Ncodon.drop('Ncodons', axis=1)
# X_95 = outlier_remover(X_Ncodon, quantile_sup=.95, features=X_nucle.columns)

"""
---y_kingdom = value of kingdom dataframe => removed
y_dnatype = value of dnatype 

y1 = kingdoms divided in prokariota eukariota and virus
y2 = kingdoms divided in phagi virus archea, bacteria, plants and animalia 
+y3 = kingdoms divided in phagi virus archea bacteria, plants, animalia-vertebrate and animalia-invertebrate 

"""

# y_kingdom = df.Kingdom
y_dnatype = df.DNAtype
y1 = first_kingdom_classifier(df)
y2 = second_kingdom_classifier(df)
y3 = test_kingdom_classifier(df)

#################################################################################
#
#                           PREPROCESSING
#
#################################################################################

scaled_X = X_nucle
scaled_X_Ncodon = X_Ncodon
minmaxscaler = pp.MinMaxScaler()

X_MMS = pd.DataFrame(minmaxscaler.fit_transform(scaled_X))
X_Codon_MMS = pd.DataFrame(minmaxscaler.fit_transform(scaled_X_Ncodon))

visualizator_2d(X_MMS, indexes=scaled_X.index, title="Min Max Scaling")
visualizator_2d(X_Codon_MMS, indexes=scaled_X.index, title="Min Max Scaling with Ncodon")

visualizator_3d(X_MMS)
visualizator_3d(X_Codon_MMS)

# ----------------------- PCA ----------------------------------------------- #

variance = []
for i in range(3, 40):
    pca_xd = PCA(n_components=i, random_state=420)
    pca_xd.fit(X_nucle)
    # = pca_xd.transform(df.drop(meta_columns, axis=1))
    variance.append(pca_xd.explained_variance_ratio_.sum())
plt.figure(1, figsize=(15, 6), constrained_layout=True)
plt.plot(np.arange(3, 40), variance, 'o')
plt.plot(np.arange(3, 40), variance, '-', alpha=0.5)
plt.xlabel('Number of dimension'), plt.ylabel('Variance')
plt.show()

pca_15d = PCA(n_components=15, random_state=420)
X_15d = pca_15d.fit_transform(X_nucle)
X_15d = pd.DataFrame(X_15d)
X_15d = pd.concat([X_15d, data_kingdom.Kingdom.reset_index()], axis=1)
plt.figure(constrained_layout=True)
plt.title("15 dimensional PCA pairplot")
sns.pairplot(data=X_15d, hue="Kingdom")
plt.show()

pca_4d = PCA(n_components=4, random_state=420)
X_4d = pca_4d.fit_transform(X_nucle)
X_4d = pd.DataFrame(X_4d)
X_4d = pd.concat([X_4d, data_kingdom.Kingdom.reset_index()], axis=1)
plt.figure(constrained_layout=True)
plt.title("4 dimensional PCA pairplot")
sns.pairplot(data=X_4d, hue="Kingdom")
plt.show()
X_4d = X_4d.drop("Kingdom", axis=1)
x_4d = X_4d
X_15d = X_15d.drop("Kingdom", axis=1)
x_15d = X_15d  # < = this little caused me some serious time lost
######################################################################################
#                           Classificationsssssssssss ---~~~~~~~~~~~~[:->~~
######################################################################################

"""
so we have: 
inputs:
X_Codon_MMS => dataset of codons and number of codons
X_MMS => dataset of codons in Min max Scaling
X_nucle => dataset of codons crude
X_4d => dataset 4 dimensional feature space
x_15d => dataset with 15 dimension 
outputs :
y_dnatype => dnatype (you don't say ? )
y1 => division in eucariote, procariote and virus
y2 => division of virus, archea, bacteria, bacteriophagi, plant and animalia
y3 => division of the class animalia in animalia vertebrate and invertebrate

"""
##########################################################################
# --------------------- SVM ---------------------------------------------#
##########################################################################
print("\n-----------------[SVM Algorithm]-----------------\n")
svc_grid = {'kernel': ['rbf', 'poly', 'linear'],  # best poly
            'degree': np.arange(2, 11, 1),  # best 4
            }
svc_estimator = SVC()
svc_cv = GridSearchCV(estimator=svc_estimator, param_grid=svc_grid,
                      scoring='f1_weighted',
                      verbose=3, n_jobs=4)

# X NUCLE Y DNA TYPE
# {'degree': 2, 'kernel': 'poly'}
# F1-Score: 0.9751315155931704
svc_cv.fit(X_nucle, y_dnatype)
print("\nX=> only frequency \t Y=> Dna type")
print("Best Params for Support Vector Classifier:\n",
      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
svc_xn_yd = svc_cv.best_params_

# X NUCLE Y 1
#  {'degree': 4, 'kernel': 'poly'}
# F1-Score: 0.937621354456861
svc_cv.fit(X_nucle, y1)
print("\nX=> only frequency \t Y=> Y1")
print("Best Params for Support Vector Classifier:\n",
      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
svc_xn_y1 = svc_cv.best_params_

# X NUCLE Y 2
# {'degree': 4, 'kernel': 'poly'}
# F1-Score: 0.9344113792349702
svc_cv.fit(X_nucle, y2)
print("\nX=> only frequency \t Y=> Y2")
print("Best Params for Support Vector Classifier:\n",
      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
svc_xn_y2 = svc_cv.best_params_

# X NUCLE Y 3
# {'degree': 4, 'kernel': 'poly'}
# F1-Score: 0.9372150944681849
svc_cv.fit(X_nucle, y3)
print("\nX=> only frequency \t Y=> Y3")
print("Best Params for Support Vector Classifier:\n",
      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
svc_xn_y3 = svc_cv.best_params_

# X_codon_MMS------------------------------
# Y DNA TYPE
#  {'degree': 2, 'kernel': 'linear'}
# F1-Score: 0.9753389707946747
svc_cv.fit(X_Codon_MMS, y_dnatype)
print("\nX=> Codon MMS \t Y=> Dna type")
print("Best Params for Support Vector Classifier:\n",
      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
svc_xc_yd = svc_cv.best_params_

# X Codon MMS Y 1
#  {'degree': 4, 'kernel': 'poly'}
# F1-Score: 0.937621354456861
svc_cv.fit(X_Codon_MMS, y1)
print("\nX=> Codon MMS \t Y=> Y1")
print("Best Params for Support Vector Classifier:\n",
      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
svc_xc_y1 = svc_cv.best_params_
# X Codon MMS Y 2
# {'degree': 4, 'kernel': 'poly'}
# F1-Score: 0.9344113792349702
svc_cv.fit(X_Codon_MMS, y2)
print("\nX=> Codon MMS \t Y=> Y2")
print("Best Params for Support Vector Classifier:\n",
      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
svc_xc_y2 = svc_cv.best_params_
# X Codon MMS Y 3
# {'degree': 4, 'kernel': 'poly'}
# F1-Score: 0.9372150944681849
svc_cv.fit(X_Codon_MMS, y3)
print("\nX=> Codon MMS \t Y=> Y3")
print("Best Params for Support Vector Classifier:\n",
      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
svc_xc_y3 = svc_cv.best_params_

# ######################### PCA SVM #######################################

# 4D PCA -----------------------------------------------------------------

# x 4d PCA Y DNAtype
# kernel : linear
# F1-Score : 0.6625530276457284
# svc_cv.fit(X_4d, y_dnatype)
# print("\nX=> 4D PCA\t Y=> dna type")
# print("Best Params for Support Vector Classifier:\n",
#      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
# svc_x4d_yd = svc_cv.best_params_
# svc_x3_yd = SVC(kernel='linear')


# X  PCA 4D Y 1
# Best Params for Support Vector Classifier:
#  {'degree': 3, 'kernel': 'poly'}
# F1-Score: 0.8725856985596273
# svc_cv.fit(X_4d, y1)
# print("\nX=> 4D PCA\t Y=> Y1")
# print("Best Params for Support Vector Classifier:\n",
#      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
# svc_x4_y1 = svc_cv.best_params_
# svc_x3_y1 = SVC(kernel='poly', degree=3)

# X  PCA 4D Y2
# Best Params for Support Vector Classifier:
# {'degree': 2, 'kernel': 'linear'}
# F1-Score: 0.9657730338179528
# svc_cv.fit(X_4d, y2)
# print("\nX=> 4D PCA\t Y=> Y2")
# print("Best Params for Support Vector Classifier:\n",
#      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
# svc_x4_y2 = svc_cv.best_params_
# svc_x3_y2 = SVC(kernel='linear')

# X  PCA 4D Y
# svc_cv.fit(X_4d, y3)
# print("\nX=> 4D PCA\t Y=> Y3")
# print("Best Params for Support Vector Classifier:\n",
#      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
# svc_x4_y3 = svc_cv.best_params_

# 15D PCA ----------------------------------------------------------------
# svc_cv.fit(X_15d, y_dnatype)
# print("\nX=> 15D PCA\t Y=> dna type")
# print("Best Params for Support Vector Classifier:\n",
#      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
# svc_x15_yd = svc_cv.best_params_

# X  PCA 4D Y
# svc_cv.fit(X_15d, y1)
# print("\nX=> 15D PCA\t Y=> Y1")
# print("Best Params for Support Vector Classifier:\n",
#      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
# svc_x15_y1 = svc_cv.best_params_
# X  PCA 15D Y1
# svc_cv.fit(X_15d, y2)
# print("\nX=> 15D PCA\t Y=> Y2")
# print("Best Params for Support Vector Classifier:\n",
#      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
# svc_x15_y2 = svc_cv.best_params_
# X PCA 15 D Y 3
# svc_cv.fit(X_15d, y3)
# print("\nX=> 15D PCA\t Y=> Y3")
# print("Best Params for Support Vector Classifier:\n",
#      svc_cv.best_params_, "\nF1-Score:", svc_cv.best_score_)
# svc_x15_y3 = svc_cv.best_params_

##########################################################################
# -----------------------------  MLP ----------------------------------- #
##########################################################################
print("\n\n\n-----------------[MLP Algorithm]-----------------\n")
mlp2_grid = {'hidden_layer_sizes': [(65, 50, 50, 50, 25, 21),
                                    (50, 50, 50, 50),
                                    (100, 90, 80, 70, 60, 50, 50, 40),
                                    (100, 50),
                                    (100,),
                                    (500,)],
             'activation': ['relu', 'logistic', 'tahn']}
mlp2_estimator = MLPClassifier(max_iter=500,
                               early_stopping=True)
mlp2_cv = mlp_cv = GridSearchCV(estimator=mlp2_estimator,
                                param_grid=mlp2_grid, cv=10,
                                verbose=1, n_jobs=4, scoring='f1_weighted')

# X MMS Y DNAtype
# {'activation': 'relu', 'hidden_layer_sizes': (100, 90, 80, 70, 60, 50, 50, 40)}
# F1-Score: 0.9886072959369262
mlp2_cv.fit(X_MMS, y_dnatype)
print("\n\nX => MMS \t Y => DNA ")
print("Best Params for Multy-Layer Perceptor Classifier:\n",
      mlp2_cv.best_params_,
      "\nF1-Score:", mlp2_cv.best_score_)
mlp_xmms_yd = mlp2_cv.best_params_

# X XMMS Y 1
# {'activation': 'relu',
# 'hidden_layer_sizes': (100, 90, 80, 70, 60, 50, 50, 40)}
# F1-Score: 0.9424801915320662
mlp2_cv.fit(X_MMS, y1)
print("\n\nX => XMMS \t Y => y1 ")
print("Best Params for Multy-Layer Perceptor Classifier:\n",
      mlp2_cv.best_params_,
      "\nF1-Score:", mlp2_cv.best_score_)
mlp_xmms_y1 = mlp2_cv.best_params_
# X NCULE Y 2
#  {'activation': 'relu',
#  'hidden_layer_sizes': (100, 90, 80, 70, 60, 50, 50, 40)}
# F1-Score: 0.9332767399087734
mlp2_cv.fit(X_MMS, y2)
print("\n\nX => XMMS \t Y => y2 ")
print("Best Params for Multy-Layer Perceptor Classifier:\n",
      mlp2_cv.best_params_,
      "\nF1-Score:", mlp2_cv.best_score_)
mlp_xmms_y2 = mlp2_cv.best_params_
# X NCULE Y 3
# {'activation': 'relu',
# 'hidden_layer_sizes': (65, 50, 50, 50, 25, 21)}
# F1-Score: 0.9880585549273851
mlp2_cv.fit(X_MMS, y3)
print("\n\nX => XMMS \t Y => 3 ")
print("Best Params for Multy-Layer Perceptor Classifier:\n",
      mlp2_cv.best_params_,
      "\nF1-Score:", mlp2_cv.best_score_)
mlp_xmms_y3 = mlp2_cv.best_params_

# X with Ncodons ----------------------------

# X MMS Y DNAtype
# {'activation': 'relu',
# 'hidden_layer_sizes': (65, 50, 50, 50, 25, 21)}
# F1-Score: 0.9881151995487742
mlp2_cv.fit(X_Codon_MMS, y_dnatype)
print("\n\nX => X_Codon_MMS \t Y => DNA ")
print("Best Params for Multy-Layer Perceptor Classifier:\n",
      mlp2_cv.best_params_,
      "\nF1-Score:", mlp2_cv.best_score_)
mlp_xmms_nc_yd = mlp2_cv.best_params_

# X XMMS Y 1
# {'activation': 'relu',
# 'hidden_layer_sizes': (100, 90, 80, 70, 60, 50, 50, 40)}
# F1-Score: 0.9348860072561898
mlp2_cv.fit(X_Codon_MMS, y1)
print("\n\nX => X_Codon_MMS \t Y => y1 ")
print("Best Params for Multy-Layer Perceptor Classifier:\n",
      mlp2_cv.best_params_,
      "\nF1-Score:", mlp2_cv.best_score_)
mlp_xmms_nc_y1 = mlp2_cv.best_params_

# X NCULE Y 2
# {'activation': 'relu',
# 'hidden_layer_sizes': (500,)}
# F1-Score: 0.9890278263817667
mlp2_cv.fit(X_Codon_MMS, y2)
print("\n\nX => X_Codon_MMS \t Y => y2 ")
print("Best Params for Multy-Layer Perceptor Classifier:\n",
      mlp2_cv.best_params_,
      "\nF1-Score:", mlp2_cv.best_score_)
mlp_xmms_nc_y2 = mlp2_cv.best_params_

# X NCULE Y 3
# {'activation': 'relu',
# 'hidden_layer_sizes': (100, 90, 80, 70, 60, 50, 50, 40)}
mlp2_cv.fit(X_Codon_MMS, y3)
print("\n\nX => X_Codon_MMS \t Y => 3 ")
print("Best Params for Multy-Layer Perceptor Classifier:\n",
      mlp2_cv.best_params_,
      "\nF1-Score:", mlp2_cv.best_score_)
mlp_xmms_nc_y3 = mlp2_cv.best_params_

##########################################################################
# --------------------- RANDOM FORESTS ----------------------------------#
##########################################################################
print("\n-----------------[Random Forest]-----------------\n")
# Random Forest
random_forest_grid = {'class_weight': [None, 'balanced', 'balanced_subsample']}
random_forest_estimator = RandomForestClassifier()
random_forest_cv = GridSearchCV(estimator=random_forest_estimator,
                                param_grid=random_forest_grid,
                                n_jobs=-1, verbose=4, scoring='f1_weighted')

# X NUCLE Y DNAtype
# {'class_weight': None}
# F1-Score: 0.9597883524422881
random_forest_cv.fit(X_nucle, y_dnatype)
print("\n\nX => X_nucle \t Y => dnatype ")
print("Best Params for Random Forest Classifier:\n",
      random_forest_cv.best_params_,
      "\nF1-Score:", random_forest_cv.best_score_)

# X NUCLE Y 1
# {'class_weight': None}
# F1-Score: 0.9172511417535351
random_forest_cv.fit(X_nucle, y1)
print("\n\nX => X_nucle \t Y => 1 ")
print("Best Params for Random Forest Classifier:\n",
      random_forest_cv.best_params_,
      "\nF1-Score:", random_forest_cv.best_score_)

# X NUCLE Y 2
# {'class_weight': 'balanced'}
# F1-Score: 0.8983703929252072
random_forest_cv.fit(X_nucle, y2)
print("\n\nX => X_nucle \t Y => 2 ")
print("Best Params for Random Forest Classifier:\n",
      random_forest_cv.best_params_,
      "\nF1-Score:", random_forest_cv.best_score_)

# X NUCLE Y 3
# {'class_weight': 'balanced_subsample'}
# F1-Score: 0.9069491582602026
random_forest_cv.fit(X_nucle, y3)
print("\n\nX => X_nucle \t Y => 3 ")
print("Best Params for Random Forest Classifier:\n",
      random_forest_cv.best_params_,
      "\nF1-Score:", random_forest_cv.best_score_)

# X Codon MMS
# X => X_Codon_MMMS 	 Y => dna
# Best Params for Random Forest Classifier:
# {'class_weight': None}
# F1-Score: 0.9562821646740923
random_forest_cv.fit(X_Codon_MMS, y_dnatype)
print("\n\nX => X_Codon_MMMS \t Y => dna ")
print("Best Params for Random Forest Classifier:\n",
      random_forest_cv.best_params_,
      "\nF1-Score:", random_forest_cv.best_score_)
# X Codon MMS Y1
# X => X_Codon_MMMS 	 Y => y1
# Best Params for Random Forest Classifier:
#  {'class_weight': None}
# F1-Score: 0.9204593999695749
#
random_forest_cv.fit(X_Codon_MMS, y1)
print("\n\nX => X_Codon_MMMS \t Y => y1")
print("Best Params for Random Forest Classifier:\n",
      random_forest_cv.best_params_,
      "\nF1-Score:", random_forest_cv.best_score_)
#
# X => X_Codon_MMMS 	 Y => y2
# Best Params for Random Forest Classifier:
#  {'class_weight': 'balanced'}
# F1-Score: 0.8999519368955529
#
random_forest_cv.fit(X_Codon_MMS, y2)
print("\n\nX => X_Codon_MMMS \t Y => y2")
print("Best Params for Random Forest Classifier:\n",
      random_forest_cv.best_params_,
      "\nF1-Score:", random_forest_cv.best_score_)

# Y3
# Best Params for Random Forest Classifier:
#  {'class_weight': 'balanced'}
# F1-Score: 0.9069848812741366
#
random_forest_cv.fit(X_Codon_MMS, y3)
print("\n\nX => X_Codon_MMMS \t Y => y3")
print("Best Params for Random Forest Classifier:\n",
      random_forest_cv.best_params_,
      "\nF1-Score:", random_forest_cv.best_score_)

#######################################################
#                   Cluster algorithms
#######################################################
inertia = []
silhouette_array = []
range_knn = np.arange(2, 15)
for n in range_knn:
    algorithm = (KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                        tol=0.0001, random_state=111, algorithm='elkan'))
    algorithm.fit_predict(X_nucle)
    labels_kmeans = algorithm.labels_
    inertia.append(algorithm.inertia_)
    silhouette_array.append(silhouette_score(X_nucle, labels_kmeans))

plt.figure(1, figsize=(15, 6), constrained_layout=True)
plt.subplot(1, 2, 1)
plt.plot(range_knn, inertia, 'o')
plt.plot(range_knn, inertia, '-', alpha=0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
plt.subplot(1, 2, 2)
plt.plot(range_knn, silhouette_array)
plt.xlabel('Number of Clusters'), plt.ylabel('Silhoutte Score')
plt.suptitle("X_nucle")
plt.show()

# X 15d
x_15d = x_15d.set_index(data_kingdom.index)
inertia = []
silhouette_array = []
range_knn = np.arange(2, 15)
for n in range_knn:
    algorithm = (KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                        tol=0.0001, random_state=111, algorithm='elkan'))
    algorithm.fit_predict(x_15d)
    labels_kmeans = algorithm.labels_
    inertia.append(algorithm.inertia_)
    silhouette_array.append(silhouette_score(x_15d, labels_kmeans))

plt.figure(1, figsize=(15, 6), constrained_layout=True)

plt.subplot(1, 2, 1)
plt.plot(range_knn, inertia, 'o')
plt.plot(range_knn, inertia, '-', alpha=0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
plt.subplot(1, 2, 2)
plt.plot(range_knn, silhouette_array)
plt.xlabel('Number of Clusters'), plt.ylabel('Silhoutte Score')
plt.suptitle("X_15D")
plt.show()

# X 4d
inertia = []
silhouette_array = []
range_knn = np.arange(2, 15)
for n in range_knn:
    algorithm = (KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                        tol=0.0001, random_state=111, algorithm='elkan'))
    algorithm.fit_predict(X_4d)
    labels_kmeans = algorithm.labels_
    inertia.append(algorithm.inertia_)
    silhouette_array.append(silhouette_score(X_4d, labels_kmeans))

plt.figure(1, figsize=(15, 6), constrained_layout=True)

plt.subplot(1, 2, 1)
plt.plot(range_knn, inertia, 'o')
plt.plot(range_knn, inertia, '-', alpha=0.5)
plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
plt.subplot(1, 2, 2)
plt.plot(range_knn, silhouette_array)
plt.xlabel('Number of Clusters'), plt.ylabel('Silhoutte Score')
plt.suptitle("X_4d")
plt.show()


# -------------------- Agglomerative Clustering -------------------------

def silhoutte_cluster(silhoutte_array, range_clusters, title):
    plt.figure(figsize=(10, 10), constrained_layout=True)
    plt.plot(range_aglomerative, silhouette_array)
    plt.xlabel('Number of Clusters'), plt.ylabel('Silhoutte Score')
    plt.suptitle(title)
    plt.show()


distances = []
silhouette_array = []
range_aglomerative = np.arange(2, 14, 1)
for n_clusters in range_aglomerative:
    aglomerativator = (AgglomerativeClustering(n_clusters=n_clusters, compute_distances=True))
    aglomerativator.fit(X_MMS)
    label_agg = aglomerativator.labels_
    distances.append(aglomerativator.distances_)
    silhouette_array.append(silhouette_score(X_MMS, label_agg, metric='euclidean'))

silhoutte_cluster(silhouette_array, range_aglomerative, title="X_MMS")
for item in range(1, len(distances), 1):
    print("avg distance:", np.mean(distances[item]))

distances = []
silhouette_array = []
for n_clusters in range_aglomerative:
    aglomerativator = (AgglomerativeClustering(n_clusters=n_clusters, compute_distances=True))
    aglomerativator.fit(x_15d)
    label_agg = aglomerativator.labels_
    distances.append(aglomerativator.distances_)
    silhouette_array.append(silhouette_score(x_15d, label_agg, metric='euclidean'))

silhoutte_cluster(silhouette_array, range_aglomerative, title="X_15d")
for item in range(1, len(distances), 1):
    print("avg distance:", np.mean(distances[item]))

distances = []
silhouette_array = []
for n_clusters in range_aglomerative:
    aglomerativator = (AgglomerativeClustering(n_clusters=n_clusters, compute_distances=True))
    aglomerativator.fit(X_4d)
    label_agg = aglomerativator.labels_
    distances.append(aglomerativator.distances_)
    silhouette_array.append(silhouette_score(X_4d, label_agg, metric='euclidean'))

silhoutte_cluster(silhouette_array, range_aglomerative, title="X_4d")
for item in range(1, len(distances), 1):
    print("avg distance:", np.mean(distances[item]))

# --------------------------- DBSCAN ---------------------------------
eps = [0.001, 0.01, 0.05, 0.1, 0.5 ]
min_sample = [1, 5, 10, 15, 20, 25]
silhouette_matrix = np.zeros((len(eps) * len(min_sample)))
n_cluster_matrix = np.zeros((len(eps) * len(min_sample)))
silhouette_matrix = silhouette_matrix.reshape(len(eps), len(min_sample))
n_cluster_matrix = n_cluster_matrix.reshape(len(eps), len(min_sample))
eps_y = 0
sample_x = 0
for eps_value in eps:
    for min_sample_value in min_sample:
        dbscanner = (DBSCAN(n_jobs=4, eps=eps_value, min_samples=min_sample_value))
        dbscanner.fit_predict(X_nucle)
        dbscanner_labels = dbscanner.labels_
        print("####################################################")
        print("eps: ", eps_value, "\tmin_sample:", min_sample_value)
        n_cluster = len(np.unique(dbscanner_labels))
        n_cluster_matrix[eps_y, sample_x] = n_cluster
        if 2 <= n_cluster < len(X_nucle):
            s_score = silhouette_score(X_nucle, dbscanner_labels)
            silhouette_matrix[eps_y, sample_x] = s_score
            print("N Clusters:", n_cluster, "\tSilhouette Score: ", s_score)
            print("\n \n")
        else:
            silhouette_matrix[eps_y, sample_x] = -1
            print("Number of cluster = 1")
        sample_x += 1
    sample_x = 0
    eps_y += 1

plt.figure(figsize=(10, 10), constrained_layout=True)
plt.title("X_nucle")
sns.heatmap(data=silhouette_matrix, annot=n_cluster_matrix,
            xticklabels=min_sample, yticklabels=eps)
plt.xlabel("min_sample")
plt.ylabel("eps")
plt.show()

# X_MMS
eps = [0.1, 0.2, 0.5, 0.7, 0.75, 1]
min_sample = [1, 5, 10, 15, 20, 25]
silhouette_matrix = np.zeros((len(eps) * len(min_sample)))
n_cluster_matrix = np.zeros((len(eps) * len(min_sample)))
silhouette_matrix = silhouette_matrix.reshape(len(eps), len(min_sample))
n_cluster_matrix = n_cluster_matrix.reshape(len(eps), len(min_sample))
eps_y = 0

for eps_value in eps:
    sample_x = 0
    for min_sample_value in min_sample:
        dbscanner = (DBSCAN(n_jobs=4, eps=eps_value, min_samples=min_sample_value))
        dbscanner.fit_predict(X_MMS)
        dbscanner_labels = dbscanner.labels_
        print("####################################################")
        print("eps: ", eps_value, "\tmin_sample:", min_sample_value)
        n_cluster = len(np.unique(dbscanner_labels))
        n_cluster_matrix[eps_y, sample_x] = n_cluster
        if 2 <= n_cluster < len(X_MMS):
            s_score = silhouette_score(X_MMS, dbscanner_labels)
            silhouette_matrix[eps_y, sample_x] = s_score
            print("N Clusters:", n_cluster, "\tSilhouette Score: ", s_score)
            print("\n \n")
        else:
            silhouette_matrix[eps_y, sample_x] = -1
            print("Number of cluster = 1")
        sample_x += 1
    eps_y += 1

plt.figure(figsize=(10, 10), constrained_layout=True)
plt.title("X_MMS")
sns.heatmap(data=silhouette_matrix, annot=n_cluster_matrix,
            xticklabels=min_sample, yticklabels=eps)
plt.xlabel("min_sample")
plt.ylabel("eps")
plt.show()

# X 15D
eps = [0.001, 0.01, 0.02, 0.05, 0.07, 0.1]
min_sample = [1, 5, 10, 15, 20, 25]
silhouette_matrix = np.zeros((len(eps) * len(min_sample)))
n_cluster_matrix = np.zeros((len(eps) * len(min_sample)))
silhouette_matrix = silhouette_matrix.reshape(len(eps), len(min_sample))
n_cluster_matrix = n_cluster_matrix.reshape(len(eps), len(min_sample))
eps_y = 0

for eps_value in eps:
    sample_x = 0
    for min_sample_value in min_sample:
        dbscanner = (DBSCAN(n_jobs=4, eps=eps_value, min_samples=min_sample_value))
        dbscanner.fit_predict(X_15d)
        dbscanner_labels = dbscanner.labels_
        print("####################################################")
        print("eps: ", eps_value, "\tmin_sample:", min_sample_value)
        n_cluster = len(np.unique(dbscanner_labels))
        n_cluster_matrix[eps_y, sample_x] = n_cluster
        if 2 <= n_cluster < len(X_15d):
            s_score = silhouette_score(X_15d, dbscanner_labels)
            silhouette_matrix[eps_y, sample_x] = s_score
            print("N Clusters:", n_cluster, "\tSilhouette Score: ", s_score)
            print("\n \n")
        else:
            silhouette_matrix[eps_y, sample_x] = -1
            print("Number of cluster = 1")
        sample_x += 1

    eps_y += 1

plt.figure(figsize=(10, 10), constrained_layout=True)
plt.title("X_15D")

sns.heatmap(data=silhouette_matrix, annot=n_cluster_matrix,
            xticklabels=min_sample, yticklabels=eps)
plt.xlabel("min_sample")
plt.ylabel("eps")
plt.show()

# X4d
eps = [0.001, 0.01, 0.015, 0.02, 0.025]
min_sample = [1, 5, 10, 15, 20, 25]
silhouette_matrix = np.zeros((len(eps) * len(min_sample)))
n_cluster_matrix = np.zeros((len(eps) * len(min_sample)))
silhouette_matrix = silhouette_matrix.reshape(len(eps), len(min_sample))
n_cluster_matrix = n_cluster_matrix.reshape(len(eps), len(min_sample))
eps_y = 0
sample_x = 0
for eps_value in eps:
    for min_sample_value in min_sample:
        dbscanner = (DBSCAN(n_jobs=4, eps=eps_value, min_samples=min_sample_value))
        dbscanner.fit_predict(X_4d)
        dbscanner_labels = dbscanner.labels_
        print("####################################################")
        print("eps: ", eps_value, "\tmin_sample:", min_sample_value)
        n_cluster = len(np.unique(dbscanner_labels))
        n_cluster_matrix[eps_y, sample_x] = n_cluster
        if 2 <= n_cluster < len(X_4d):
            s_score = silhouette_score(X_MMS, dbscanner_labels)
            silhouette_matrix[eps_y, sample_x] = s_score
            print("N Clusters:", n_cluster, "\tSilhouette Score: ", s_score)
            print("\n \n")
        else:
            silhouette_matrix[eps_y, sample_x] = -1
            print("Number of cluster = 1")
        sample_x += 1
    sample_x = 0
    eps_y += 1

plt.figure(figsize=(10, 10), constrained_layout=True)
plt.title("X_4d")
sns.heatmap(data=silhouette_matrix, annot=n_cluster_matrix,
            xticklabels=min_sample, yticklabels=eps)
plt.xlabel("min_sample")
plt.ylabel("eps")
plt.show()

##################################################################################
#                               Conclusion
################################################################################


X_nucle, X_nucle_test, X_MMS, X_MMS_test, X_Codon_MMS, X_Codon_MMS_test, \
X_4d, X_4d_test, X_15d, X_15d_test, \
y_dnatype, y_dnatype_test, y1, y1_test, \
y2, y2_test, y3, y3_test = train_test_split(X_nucle, X_MMS, X_Codon_MMS,
                                            X_4d, X_15d,
                                            y_dnatype, y1,
                                            y2, y3,
                                            test_size=0.20, random_state=420)
# --------------------- SVM Classifiers ----------
print("#-----------SVC-----------#")
print("SVC for X_NUCE")
print("\nY DNA_TYPE")
svc_x1_yd = SVC(kernel='poly', degree=2)
t0 = time.time()
svc_x1_yd.fit(X_nucle, y_dnatype)
t1 = time.time()
y_hat = svc_x1_yd.predict(X_nucle_test)
f_score = f1_score(y_dnatype_test, y_hat, average="weighted")
p_score = precision_score(y_dnatype_test, y_hat, average="weighted")
a_score = accuracy_score(y_dnatype_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y_dnatype_test, y_hat, "Y DNA type X_nucle svm")

print("\nY 1")
svc_x1_y1 = SVC(kernel='poly', degree=4)
t0 = time.time()
svc_x1_y1.fit(X_nucle, y1)
t1 = time.time()
t_tot = t1 - t0
y_hat = svc_x1_y1.predict(X_nucle_test)
f_score = f1_score(y1_test, y_hat, average="weighted")
p_score = precision_score(y1_test, y_hat, average="weighted")
a_score = accuracy_score(y1_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y1_test, y_hat, "Y 1 X_nucle svm")

print("\nY 2")
svc_x1_y2 = SVC(kernel='poly', degree=2)
t0 = time.time()
svc_x1_y2.fit(X_nucle, y2)
t1 = time.time()
y_hat = svc_x1_y2.predict(X_nucle_test)
f_score = f1_score(y2_test, y_hat, average="weighted")
p_score = precision_score(y2_test, y_hat, average="weighted")
a_score = accuracy_score(y2_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y2_test, y_hat, "Y 2 X_nucle svm")

print("\nY 3")
svc_x1_y3 = SVC(kernel='poly', degree=4)
t0 = time.time()
svc_x1_y3.fit(X_nucle, y3)
t1 = time.time()
y_hat = svc_x1_y3.predict(X_nucle_test)
f_score = f1_score(y3_test, y_hat, average="weighted")
p_score = precision_score(y3_test, y_hat, average="weighted")
a_score = accuracy_score(y3_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y3_test, y_hat, "Y 3 X_nucle svm")

print("\n\nSVC for X_Codon_MMS")
print("\nY DNA type")
svc_x2_yd = SVC(kernel='linear')
t0 = time.time()
svc_x2_yd.fit(X_Codon_MMS, y_dnatype)
t1 = time.time()
y_hat = svc_x2_yd.predict(X_Codon_MMS_test)
f_score = f1_score(y_dnatype_test, y_hat, average="weighted")
p_score = precision_score(y_dnatype_test, y_hat, average="weighted")
a_score = accuracy_score(y_dnatype_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y_dnatype_test, y_hat, "Y DNA type X_Codon_MMS svm")

print("\nY1")
svc_x2_y1 = SVC(kernel='poly', degree=4)
t0 = time.time()
svc_x2_y1.fit(X_Codon_MMS, y1)
t1 = time.time()
y_hat = svc_x2_y1.predict(X_Codon_MMS_test)
f_score = f1_score(y1_test, y_hat, average="weighted")
p_score = precision_score(y1_test, y_hat, average="weighted")
a_score = accuracy_score(y1_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y1_test, y_hat, "Y 1 X_Codon_MMS svm")

print("\nY2")
svc_x2_y2 = SVC(kernel='poly', degree=4)
t0 = time.time()
svc_x2_y2.fit(X_Codon_MMS, y2)
t1 = time.time()
y_hat = svc_x2_y2.predict(X_Codon_MMS_test)
f_score = f1_score(y2_test, y_hat, average="weighted")
p_score = precision_score(y2_test, y_hat, average="weighted")
a_score = accuracy_score(y2_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y2_test, y_hat, "Y 2 X_Codon_MMS svm")

print("\nY 3")
svc_x2_y3 = SVC(kernel='poly', degree=4)
t0 = time.time()
svc_x2_y3.fit(X_Codon_MMS, y3)
t1 = time.time()
y_hat = svc_x2_y3.predict(X_Codon_MMS_test)
f_score = f1_score(y3_test, y_hat, average="weighted")
p_score = precision_score(y3_test, y_hat, average="weighted")
a_score = accuracy_score(y3_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y3_test, y_hat, "Y 3 X_Codon_MMS svm")

# --------------------- MLP Classifiers ----------------------#
print("\n\n#-----------MLP-----------#")
print("MLP for X_MMS")
print("\nY DNAtype")
mlp_x1_yd = MLPClassifier(hidden_layer_sizes=(100, 90, 80, 70, 60, 50, 50, 40), activation='relu')
t0 = time.time()
mlp_x1_yd.fit(X_MMS, y_dnatype)
t1 = time.time()
y_hat = mlp_x1_yd.predict(X_MMS_test)
f_score = f1_score(y_dnatype_test, y_hat, average="weighted")
p_score = precision_score(y_dnatype_test, y_hat, average="weighted")
a_score = accuracy_score(y_dnatype_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y_dnatype_test, y_hat, "Y DNA type X_MMS mlp")

print("\nY1")
mlp_x1_y1 = MLPClassifier(hidden_layer_sizes=(100, 90, 80, 70, 60, 50, 50, 40), activation='relu')
t0 = time.time()
mlp_x1_y1.fit(X_MMS, y1)
t1 = time.time()
y_hat = mlp_x1_y1.predict(X_MMS_test)
f_score = f1_score(y1_test, y_hat, average="weighted")
p_score = precision_score(y1_test, y_hat, average="weighted")
a_score = accuracy_score(y1_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y1_test, y_hat, "Y 1 X_MMS mlp")

print("\nY 2")
mlp_x1_y2 = MLPClassifier(hidden_layer_sizes=(100, 90, 80, 70, 60, 50, 50, 40), activation='relu')
t0 = time.time()
mlp_x1_y2.fit(X_MMS, y2)
t1 = time.time()
y_hat = mlp_x1_y2.predict(X_MMS_test)
f_score = f1_score(y2_test, y_hat, average="weighted")
p_score = precision_score(y2_test, y_hat, average="weighted")
a_score = accuracy_score(y2_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y2_test, y_hat, "Y 2 X_MMS mlp")

print("\nY 3")
mlp_x1_y3 = MLPClassifier(hidden_layer_sizes=(65, 50, 50, 50, 25, 21),
                          activation='relu', max_iter=500)
t0 = time.time()
mlp_x1_y3.fit(X_MMS, y3)
t1 = time.time()
y_hat = mlp_x1_y3.predict(X_MMS_test)
f_score = f1_score(y3_test, y_hat, average="weighted")
p_score = precision_score(y3_test, y_hat, average="weighted")
a_score = accuracy_score(y3_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y3_test, y_hat, "Y 3 X_MMS mlp")

print("\nMLP for Codon_MMS")
print("Y DNAtype")
mlp_x2_yd = MLPClassifier(hidden_layer_sizes=(65, 50, 50, 50, 25, 21), activation='relu')
t0 = time.time()
mlp_x2_yd.fit(X_Codon_MMS, y_dnatype)
t1 = time.time()
y_hat = mlp_x2_yd.predict(X_Codon_MMS_test)
f_score = f1_score(y_dnatype_test, y_hat, average="weighted")
p_score = precision_score(y_dnatype_test, y_hat, average="weighted")
a_score = accuracy_score(y_dnatype_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y_dnatype_test, y_hat, "Y DNA type X_Codon_MMS mlp")

print("\nY1")
mlp_x2_y1 = MLPClassifier(hidden_layer_sizes=(100, 90, 80, 70, 60, 50, 50, 40), activation='relu')
t0 = time.time()
mlp_x2_y1.fit(X_Codon_MMS, y1)
t1 = time.time()
y_hat = mlp_x2_y1.predict(X_Codon_MMS_test)
f_score = f1_score(y1_test, y_hat, average="weighted")
p_score = precision_score(y1_test, y_hat, average="weighted")
a_score = accuracy_score(y1_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y1_test, y_hat, "Y 1 X_Codon_MMS mlp")

print("\nY 2")
mlp_x2_y2 = MLPClassifier(hidden_layer_sizes=(500,),
                          activation='relu', max_iter=500)
t0 = time.time()
mlp_x2_y2.fit(X_Codon_MMS, y2)
t1 = time.time()
y_hat = mlp_x2_y2.predict(X_Codon_MMS_test)
f_score = f1_score(y2_test, y_hat, average="weighted")
p_score = precision_score(y2_test, y_hat, average="weighted")
a_score = accuracy_score(y2_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y2_test, y_hat, "Y 2 X_Codon_MMS mlp")

print("\nY 2 -- Deep")
mlp_x2_y2_doubt = MLPClassifier(hidden_layer_sizes=(100, 90, 80, 70, 60, 50, 50, 40), activation='relu')
t0 = time.time()
mlp_x2_y2_doubt.fit(X_Codon_MMS, y2)
t1 = time.time()
y_hat = mlp_x2_y2_doubt.predict(X_Codon_MMS_test)
f_score = f1_score(y2_test, y_hat, average="weighted")
p_score = precision_score(y2_test, y_hat, average="weighted")
a_score = accuracy_score(y2_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y2_test, y_hat, "Y 2 X_Codon_MMS mlp deep")

print("\nY 3")
mlp_x2_y3 = MLPClassifier(hidden_layer_sizes=(100, 90, 80, 70, 60, 50, 50, 40), activation='relu')
t0 = time.time()
mlp_x2_y3.fit(X_Codon_MMS, y3)
t1 = time.time()
y_hat = mlp_x2_y3.predict(X_Codon_MMS_test)
f_score = f1_score(y3_test, y_hat, average="weighted")
p_score = precision_score(y3_test, y_hat, average="weighted")
a_score = accuracy_score(y3_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y3_test, y_hat, "Y 3 X_Codon_MMS mlp")

# -------------------- Random Forests --------------
print("#-----------Random Forest-----------#")
print("Random Forest for X_NUCE")
print("Y DNAtype")
random_x1_yd = RandomForestClassifier(class_weight=None)
t0 = time.time()
random_x1_yd.fit(X_nucle, y_dnatype)
t1 = time.time()
y_hat = random_x1_yd.predict(X_nucle_test)
f_score = f1_score(y_dnatype_test, y_hat, average="weighted")
p_score = precision_score(y_dnatype_test, y_hat, average="weighted")
a_score = accuracy_score(y_dnatype_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y_dnatype_test, y_hat, "Y dnatype X_nucle random forest")

print("\nY1")
random_x1_y1 = RandomForestClassifier()
t0 = time.time()
random_x1_y1.fit(X_nucle, y1)
t1 = time.time()
y_hat = random_x1_y1.predict(X_nucle_test)
f_score = f1_score(y1_test, y_hat, average="weighted")
p_score = precision_score(y1_test, y_hat, average="weighted")
a_score = accuracy_score(y1_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y1_test, y_hat, "Y 1 X_nucle random forest")

print("\nY 2")
random_x1_y2 = RandomForestClassifier(class_weight='balanced')
t0 = time.time()
random_x1_y2.fit(X_nucle, y2)
t1 = time.time()
y_hat = random_x1_y2.predict(X_nucle_test)
f_score = f1_score(y2_test, y_hat, average="weighted")
p_score = precision_score(y2_test, y_hat, average="weighted")
a_score = accuracy_score(y2_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y2_test, y_hat, "Y 2 X_nucle random forest")

print("\nY 3")
random_x1_y3 = RandomForestClassifier(class_weight='balanced_subsample')
t0 = time.time()
random_x1_y3.fit(X_nucle, y3)
t1 = time.time()
y_hat = random_x1_y3.predict(X_nucle_test)
f_score = f1_score(y3_test, y_hat, average="weighted")
p_score = precision_score(y3_test, y_hat, average="weighted")
a_score = accuracy_score(y3_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y3_test, y_hat, "Y 3 X_nucle random forest")

print("Random forest for X_Codon_MMS")
print("\nY DNAtype")
random_x2_yd = RandomForestClassifier()
t0 = time.time()
random_x2_yd.fit(X_Codon_MMS, y_dnatype)
t1 = time.time()
y_hat = random_x2_yd.predict(X_Codon_MMS_test)
f_score = f1_score(y_dnatype_test, y_hat, average="weighted")
p_score = precision_score(y_dnatype_test, y_hat, average="weighted")
a_score = accuracy_score(y_dnatype_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y_dnatype_test, y_hat, "Y DNAtype X_Codon_MMS random forest")

print("\nY1")
random_x2_y1 = RandomForestClassifier()
t0 = time.time()
random_x2_y1.fit(X_Codon_MMS, y1)
t1 = time.time()
y_hat = random_x2_y1.predict(X_Codon_MMS_test)
f_score = f1_score(y1_test, y_hat, average="weighted")
p_score = precision_score(y1_test, y_hat, average="weighted")
a_score = accuracy_score(y1_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y1_test, y_hat, "Y 1 X_Codon_MMS random forest")

print("\nY 2")
random_x2_y2 = RandomForestClassifier()
t0 = time.time()
random_x2_y2.fit(X_Codon_MMS, y2)
t1 = time.time()
y_hat = random_x2_y2.predict(X_Codon_MMS_test)
f_score = f1_score(y2_test, y_hat, average="weighted")
p_score = precision_score(y2_test, y_hat, average="weighted")
a_score = accuracy_score(y2_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y2_test, y_hat, "Y 2 X_Codon_MMS random forest")

print("\nY 3")
random_x2_y3 = RandomForestClassifier()
t0 = time.time()
random_x2_y3.fit(X_Codon_MMS, y3)
t1 = time.time()
y_hat = random_x2_y3.predict(X_Codon_MMS_test)
f_score = f1_score(y3_test, y_hat, average="weighted")
p_score = precision_score(y3_test, y_hat, average="weighted")
a_score = accuracy_score(y3_test, y_hat)
print("Fit time:", (t1 - t0))
print("F-score   :", f_score)
print("Accuracy  :", a_score)
print("Precision :", p_score)
ROC_plotting(y3_test, y_hat, "Y 3 X_Codon_MMS random forest")

##################################################################################################################
#                           Conclusion Clustering
##################################################################################################################
X_nucle = pd.concat([X_nucle, X_nucle_test])
X_15d = pd.concat([X_15d_test, X_15d_test])
X_MMS = pd.concat([X_MMS, X_MMS_test])
X_4d = pd.concat([X_4d, X_4d_test])

######################################################
#                   KMeans
######################################################
labels = []
km_8 = (
    KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111, algorithm='elkan'))
km_8.fit_transform(X_nucle)
labels.append(km_8.labels_)
km_8.fit_transform(X_MMS)
labels.append(km_8.labels_)
km_8.fit_transform(X_4d)
labels.append(km_8.labels_)
km_8.fit_transform(X_nucle)
labels.append(km_8.labels_)
visualizator_2d_clustering(labels, "KMeans n_clustering = 8")

km_2 = (
    KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111, algorithm='elkan'))
labels = []
km_2.fit_predict(X_nucle)
labels.append(km_2.labels_)
km_2.fit_predict(X_MMS)
labels.append(km_2.labels_)
km_2.fit_predict(X_4d)
labels.append(km_2.labels_)
km_2.fit_predict(X_nucle)
labels.append(km_2.labels_)
visualizator_2d_clustering(labels, "KMeans n_clustering = 2")
kn_3 = (
    KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=111, algorithm='elkan'))

#####################################################
#                   Agg
####################################################
km_2 = (AgglomerativeClustering(n_clusters=4))
labels = []
km_2.fit_predict(X_nucle)
labels.append(km_2.labels_)
km_2.fit_predict(X_MMS)
labels.append(km_2.labels_)
km_2.fit_predict(X_4d)
labels.append(km_2.labels_)
km_2.fit_predict(X_nucle)
labels.append(km_2.labels_)
visualizator_2d_clustering(labels, "Agglomerative n_clustering = 4")

####################################################
#                  Dbscan
####################################################


labels = []
db_XN = (DBSCAN(n_jobs=4, eps=0.05, min_samples=25))
db_XN.fit_predict(X_nucle)
labels.append(db_XN.labels_)

db_XM = (DBSCAN(n_jobs=4, eps=0.5, min_samples=10))
db_XM.fit_predict(X_MMS)

labels.append(db_XM.labels_)

db_X4 = (DBSCAN(n_jobs=4, eps=0.015, min_samples=15))
db_X4.fit_predict(X_4d)

labels.append(db_X4.labels_)

db_X15 = (DBSCAN(n_jobs=4, eps=0.03, min_samples=9))
db_X15.fit_predict(X_15d)
labels.append(db_X15.labels_)

visualizator_2d_clustering(labels, title="DBSCAN", palette=cm.gist_rainbow)
visualizator_2d(X_nucle, title="Dataset labels")

visualizator_2d(X_nucle, labels=db_XN.labels_)
visualizator_2d(X_MMS, labels=db_XM.labels_)
visualizator_2d(X_4d, labels=db_X4.labels_)
visualizator_2d(X_15d, labels=db_X15.labels_)
