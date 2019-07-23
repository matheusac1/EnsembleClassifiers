from sklearn.model_selection import KFold
from sklearn import tree
import pandas as pd
import sklearn.metrics as metrics
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

dados = pd.read_csv('results_corr_065_sem_modulo_mensal_sempolo_com_qtde_links_ajustado_sem_ruido.csv',index_col=0)

#shuffle the data 
from sklearn.utils import shuffle
dados = shuffle(dados)

X = pd.DataFrame()

X['modularity'] = dados['modularity']
X['global_average_link_distance'] = dados['global_average_link_distance']
X['eigenvector'] = dados['eigenvector']
#X['eccentricity'] = dados['eccentricity']
X['transitivity'] = dados['transitivity']
X['average_path_length'] = dados['average_path_length']
X['coreness'] = dados['coreness']
#X['links'] = dados['links']
X['grauMedio'] = dados['grauMedio']

Y = np.asarray(dados['flag'])
X = np.asarray(X)

kf = KFold(n_splits=10)#divide o dataset em 10 partes 9 p/ treino e 1 teste

a = 0
f = 0
p = 0
r = 0
i = 0

for train_index, test_index in kf.split(X):
    i+=1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    clf = BaggingClassifier(max_samples=0.5, max_features=0.5, n_estimators=200).fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    a += clf.score(X_test,y_test)
    f += metrics.f1_score(y_test, y_pred,average='micro')
    p += metrics.precision_score(y_test, y_pred, average='micro')
    r += metrics.recall_score(y_test, y_pred, average='micro')

average_accuracy = a/i
average_f1_score = f/i
average_precision = p/i
average_recall = r/i

print('Accuracy: ')
print(average_accuracy)

print('F1 - Score:')
print(average_f1_score)

print('Precision:')
print(average_precision)

print('Recall:')
print(average_recall)
