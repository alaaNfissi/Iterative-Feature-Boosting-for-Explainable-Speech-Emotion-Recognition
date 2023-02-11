import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
import pandas as pd
plt.rcParams['figure.figsize'] = (21,15)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os
from tqdm.notebook import tqdm
import parselmouth
import librosa
import librosa.display
import scipy
import random
import shap
random.seed(123)

dataset_name = 'tess'

df = pd.read_csv(f'../Data_exploration/all_handcrafted_data_{dataset_name}.csv')
df = df[df['source'] == 'TESS']
df.drop(columns=['source', 'path'],inplace=True)
df.dropna(inplace=True)
df = df.loc[:,~df.columns.duplicated()].copy()
df = df.loc[:,~df.apply(lambda x: x.duplicated(),axis=1).all()].copy()
df.info()

df = df.replace(np.nan, 0)
#print(df[df.isna().any(axis=1)])

print(df.head(10))

pot_cols_list = set()
cols_number = 10
all_cols = list(df.columns.unique())
all_cols.remove('class')
saved_cols = []
saved_explained_variances = []
dic_explained_variances = {}
for i in tqdm(range(100000)):
    pot_cols = random.sample(all_cols, k=cols_number)
    print(pot_cols)
    if not any(len(set(pot_cols) & set(elem)) == cols_number for elem in pot_cols_list):
        df_aux = df[pot_cols]
        X = df_aux
        Xs = StandardScaler().fit_transform(X)
        Xcols = X.columns
        X = pd.DataFrame(Xs)
        X.columns = Xcols
        pca = PCA()
        Z = pca.fit_transform(X)
        Lambda = pca.explained_variance_ 
        explained_variance = Lambda/sum(Lambda)
        #print(explained_variance[0] + explained_variance[1])
        if explained_variance[0] + explained_variance[1] >= 0.8:
            saved_cols.append(pot_cols)
            pot_cols_list.add(tuple(pot_cols))
            saved_explained_variances.append(explained_variance[0] + explained_variance[1])
            
optimal_cols = saved_cols[saved_explained_variances.index(max(saved_explained_variances))]
print(f'Optimal features combination:{optimal_cols}')
print(f'Best features combinations:{saved_cols}')

df_aux = df[optimal_cols]
X = df_aux
Xs = StandardScaler().fit_transform(X)
Xcols = X.columns
X = pd.DataFrame(Xs)
X.columns = Xcols
pca = PCA()
Z = pca.fit_transform(X)

dic_pca = {}
for pot_cols in tqdm(saved_cols):
    df_aux = df[pot_cols]
    X = df_aux
    Xs = StandardScaler().fit_transform(X)
    Xcols = X.columns
    X = pd.DataFrame(Xs)
    X.columns = Xcols
    pca = PCA()
    Z = pca.fit_transform(X)
    Z = pca.inverse_transform(Z)
    dic_pca.update({f'PC_{saved_cols.index(pot_cols)}_1':Z[:,0], f'PC_{saved_cols.index(pot_cols)}_2':Z[:,1]})
data_pca = pd.DataFrame(dic_pca)

data_pca['class'] = list(df['class'])

df = data_pca

df.head()

df.info()

df[df.isna().any(axis=1)]

print("Number of duplicated rows is: ", df.duplicated().sum())

print("Number of rows with NaNs is: ", df.isna().any(axis=1).sum())

sns.pairplot(df[optimal_cols+['class']], hue='class')
plt.show()

y = df['class']
y.value_counts().plot(kind='pie')
plt.ylabel('')
plt.show()

print('Data Matrix')

X = df.drop(columns=['class'])
X.head(10)

X.describe().transpose()

print('Standardize the Data')

Xs = StandardScaler().fit_transform(X)
Xcols = X.columns
X = pd.DataFrame(Xs)
X.columns = Xcols
X.head(10)

X.describe().transpose()

#Observations and variables

observations = list(df.index)
variables = list(df.columns)

#Box and Whisker Plots

ax = plt.figure()
ax = sns.boxplot(data=X, orient="v", palette="Set2")
ax.set_xticklabels(ax.get_xticklabels(),rotation=45);

# Use swarmplot() or stripplot to show the datapoints on top of the boxes:
#plt. figure()
ax = plt.figure()    
ax = sns.boxplot(data=X, orient="v", palette="Set2")
ax = sns.stripplot(data=X, color=".25") 
ax.set_xticklabels(ax.get_xticklabels(),rotation=45);

#Correlation Matrix

ax = sns.heatmap(X.corr(), cmap='RdYlGn_r', linewidths=0.5, annot=True, cbar=False, square=True)
plt.yticks(rotation=0)
ax.tick_params(labelbottom=False,labeltop=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=0);


import matplotlib.cm as cm
pca = PCA()
Z = pca.fit_transform(X)

plt.figure()
idx_dic = {}
colors = cm.rainbow(np.linspace(0, 1, len(df['class'].unique())))
for i in df['class'].unique():
    idx_dic.update({f'idx_{i}': np.where(y == i)})
    plt.scatter(Z[idx_dic.get(f'idx_{i}'),0], Z[idx_dic.get(f'idx_{i}'),1], c=[colors[list(df['class'].unique()).index(i)]], label=i)
plt.legend()
plt.xlabel('$Z_1$')
plt.ylabel('$Z_2$')

#Eigenvectors

A = pca.components_.T 

plt.scatter(A[:,0],A[:,1],c='r')
plt.xlabel('$A_1$')
plt.ylabel('$A_2$')
for label, x, y in zip(variables, A[:, 0], A[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-2, 2), textcoords='offset points', ha='right', va='bottom')
    
plt.scatter(A[:, 0],A[:, 1], marker='o', c=A[:, 2], s=A[:, 3]*500, cmap=plt.get_cmap('Spectral'))
plt.xlabel('$A_1$')
plt.ylabel('$A_2$')
for label, x, y in zip(variables,A[:, 0],A[:, 1]):
    plt.annotate(label,xy=(x, y), xytext=(-20, 20),
    textcoords='offset points', ha='right', va='bottom',
    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
#Scree plot

#Eigenvalues
Lambda = pca.explained_variance_ 

#Scree plot
x = np.arange(len(Lambda)) + 1
plt.plot(x,Lambda/sum(Lambda), 'ro-', lw=3)
plt.xticks(x, [""+str(i) for i in x], rotation=0)
plt.xlabel('Number of components')
plt.ylabel('Explained variance') 


#Explained Variance

ell = pca.explained_variance_ratio_
ind = np.arange(len(ell))
plt.bar(ind, ell, align='center', alpha=0.5)
plt.plot(np.cumsum(ell))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')


from pca import pca
# Initialize and keep all PCs
model = pca()
# Fit transform
out = model.fit_transform(X)

print(out['variance_ratio'])

model.plot();

model.biplot(label=False, legend=False)

#Classification

#For Google Colab only
from pycaret.utils import enable_colab 
enable_colab()

data = df.sample(frac=0.9, random_state=786)
data_unseen = df.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

from pycaret.classification import *
clf = setup(data=data, target='class', train_size=0.7, session_id=123, data_split_stratify=True)

#Comparing All Models

#show the best models and their statistics
best_model = compare_models()

best_model

# Tune hyperparameters with scikit-learn (default)
tuned_best_model = tune_model(best_model)

predict_model(tuned_best_model)

tuned_best_model

evaluate_model(tuned_best_model)

tuned_et_pca = tuned_best_model

explainer = shap.TreeExplainer(tuned_et_pca)
X = df.drop('class', axis=1)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)

shap.summary_plot(shap_values[1], X)

interpret_model(tuned_et_pca, plot='reason', observation=12)

interpret_model(tuned_et_pca, plot='reason')

print('Done!')