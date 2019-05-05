

import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

df = pd.read_csv('/home/datascience/Desktop/SpamFiltering/features.csv', sep=',', lineterminator='\n',
                 na_values=['-'])

columns = list(df.columns.values)
words = columns[:]
words.remove("is_spam")
words.remove("file_name")
words.remove("ind")

df_spam = pd.DataFrame()
df_spam["is_spam"] = df["is_spam"]
df.drop('is_spam', axis=1, inplace=True)
df.drop('file_name', axis=1, inplace=True)
df.drop('ind', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df,df_spam, train_size = 0.6)

# 25 min
neigh = KNeighborsClassifier(n_neighbors=3) 
neigh.fit(X_train, y_train) 
neighResult = neigh.predict(X_test)
print "KNN results"
print "confusion matrix"
print confusion_matrix(y_test, neighResult)
print "precision recall scores"
print precision_recall_fscore_support(y_test,neighResult,pos_label=0)

