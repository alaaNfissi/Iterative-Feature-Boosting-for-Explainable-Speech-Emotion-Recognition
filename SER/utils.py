import numpy as np
print("Numpy version: ", np.__version__)
import pandas as pd
import os
from tqdm.notebook import tqdm
import parselmouth
import shap
from pycaret.classification import *
from pca import pca
warnings.filterwarnings('ignore')



def select_feature_combinations(df, iterations):
    pot_cols_list = set()
    cols_number = 10
    all_cols = list(df.columns.unique())
    all_cols.remove('class')
    saved_cols = []
    saved_explained_variances = []
    dic_explained_variances = {}
    for i in tqdm(range(iterations)):
        if len(all_cols) > cols_number:
            pot_cols = random.sample(all_cols, k=cols_number)
        else:
            pot_cols = all_cols
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
    return saved_cols, pot_cols_list, saved_explained_variances



def boosted_dataset_construction(saved_cols, df):
    dic_pca = {}
    for pot_cols in tqdm(saved_cols):
        df_aux = df[pot_cols]
        #print(df_aux.shape)
        X = df_aux
        Xs = StandardScaler().fit_transform(X)
        Xcols = X.columns
        X = pd.DataFrame(Xs)
        X.columns = Xcols
        pca = PCA()
        Z = pca.fit_transform(X)
        Z = pca.inverse_transform(Z)
        #print(Z.shape)
        dic_pca.update({f'PC_{saved_cols.index(pot_cols)}_1':Z[:,0], f'PC_{saved_cols.index(pot_cols)}_2':Z[:,1]})
    data_pca = pd.DataFrame(dic_pca)
    data_pca['class'] = list(df['class'])
    return data_pca

def flatten(l):
    return [item for sublist in l for item in sublist]