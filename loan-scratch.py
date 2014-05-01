__author__ = 'dan'

'''
Playground file for trying things out...
'''

import os
import sys
import xml.dom.minidom as xml
import pandas as pd
import numpy as np
import random
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
import sklearn as sl
from sklearn import neighbors
from sklearn.preprocessing import Imputer
from sklearn import preprocessing as pre
from sklearn import linear_model
import sklearn.neural_network

def mae(y_pred, y_act):
    return (np.abs(y_act - y_pred).sum() * 1.0)/len(y_pred)



def main(in_dir, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = pd.read_table(in_dir + '/full/train_v2.csv', sep=',')
    dfl = df[df['loss'] > 0]


    # output stats about rows
    # dfd = pd.DataFrame(df.describe())
    # dfo = pd.DataFrame(data=np.zeros((22,len(df.columns))), columns=df.columns)
    # for col in df.columns:
    #     dfg = df.groupby(col).agg({col:np.size, 'loss':np.count_nonzero})
    #     l = len(dfg)
    #     dfo[col][0] = l
    #     idx_len = 10
    #     if len(dfg.index) < 10:
    #         idx_len = len(dfg.index)
    #     dfo[col][1:idx_len+1] = dfg.index[0:idx_len]
    #     # print(str(dfo[col][12:idx_len+12]))
    #     # print(str(dfg[0:idx_len]))
    #     dfo[col][12:idx_len+12] = dfg[col][0:idx_len]
    #     print(col + "|" + str(l))
    #     dfg['ratio'] = dfg.apply(lambda x: x['loss']/x[col], axis=1)
    #
    #     if l <= 25:
    #         # outbout all counts and losses
    #         dfg.to_csv(out_dir + '/detail-' + col + '.csv')
    #
    #     if l <= 100 and dfg['ratio'].std() > (2 * dfg['ratio'].mean() ):
    #         dfg.to_csv(out_dir + '/highvar-' + col + '.csv')
    #
    # dfd.to_csv(out_dir + '/desc.csv')
    # dfo.to_csv(out_dir + '/dupes.csv')
    # dfx = dfd.append(dfo)
    # dfx.to_csv(out_dir + '/combo.csv')

    # f,p = feature_selection.f_regression(df.values[:,1:-4],df['loss'])
    # print (str(f))
    # print (str(p))
    # np.savetxt(out_dir + '/f.csv',np.reshape(f, (1,len(f))), delimiter=',')
    # np.savetxt(out_dir + '/p.csv',np.reshape(p, (1,len(p))), delimiter=',')


    # DEC hunting for best model on the default size regression
    excludes = ['id','f776','f777','f778','loss']
    X2 = dfl.drop(excludes, axis=1)
    y2 = dfl['loss']
    imp = Imputer()
    imp.fit(X2)
    X2 = imp.transform(X2)
    X2 = pre.StandardScaler().fit_transform(X2)

    # reduce
    #redb =  LinearSVC(C=1, penalty="l1", dual=False, verbose = 1)
    redb = sl.feature_selection.SelectKBest(sl.feature_selection.f_classif, 50)
    X2 = redb.fit(X2, y2).transform(X2)

    # print ('reducing X2 to # cols:' + str(m))
    # redb = sl.feature_selection.SelectKBest(sl.feature_selection.f_classif, m)
    # X2 = redb.fit(X2, y2).transform(X2)
    print("regression training set size=" + str(X2.shape))

    X_train, X_test, y_train, y_test = sl.cross_validation.train_test_split(X2, y2)

    # print ('linear SVC')
    # reg = sl.linear_model.LinearSVC()
    # z = reg.fit(X_train, y_train).predict(X_test)
    # print ("regression fit score: " + str(reg.score(X_train, y_train)))
    # print ('R2: ' + str(sl.metrics.r2_score(y_test, z)))
    # print ('conf matrix 2: ' + str(sl.metrics.confusion_matrix(y_test, z)))
    # print ('class report 2: ' + str(sl.metrics.classification_report(y_test, z)))

    print ('LinearRegression')
    reg = linear_model.LinearRegression()
    z = reg.fit(X_train, y_train).predict(X_test)
    print ("regression fit score: " + str(reg.score(X_train, y_train)))
    print ('R2: ' + str(sl.metrics.r2_score(y_test, z)))
    print ('mae: ' + str(mae(y_test, z)))

    print ('Ridge')
    reg = linear_model.Ridge()
    z = reg.fit(X_train, y_train).predict(X_test)
    print ("regression fit score: " + str(reg.score(X_train, y_train)))
    print ('R2: ' + str(sl.metrics.r2_score(y_test, z)))
    print ('mae: ' + str(mae(y_test, z)))

    # print ('BernoulliRBM')
    # reg = sl.neural_network.BernoulliRBM()
    # z = reg.fit(X_train, y_train).predict(X_test)
    # print ("regression fit score: " + str(reg.score(X_train, y_train)))
    # print ('R2: ' + str(sl.metrics.r2_score(y_test, z)))
    # print ('mae: ' + str(mae(y_test, z)))


    print ('SVR 1')
    reg = svm.SVR(C=1, epsilon=0.1, kernel='rbf', degree=3)
    z = reg.fit(X_train, y_train).predict(X_test)
    print ("regression fit score: " + str(reg.score(X_train, y_train)))
    print ('R2: ' + str(sl.metrics.r2_score(y_test, z)))
    print ('mae: ' + str(mae(y_test, z)))
    # print ('conf matrix 2: ' + str(sl.metrics.confusion_matrix(y_test, z)))
    # print ('class report 2: ' + str(sl.metrics.classification_report(y_test, z)))

    print ('SVR 2')
    reg = svm.SVR(C=10, epsilon=0.1, kernel='rbf', degree=3)
    z = reg.fit(X_train, y_train).predict(X_test)
    print ("regression fit score: " + str(reg.score(X_train, y_train)))
    print ('R2: ' + str(sl.metrics.r2_score(y_test, z)))
    print ('mae: ' + str(mae(y_test, z)))
    # print ('conf matrix 2: ' + str(sl.metrics.confusion_matrix(y_test, z)))
    # print ('class report 2: ' + str(sl.metrics.classification_report(y_test, z)))

    print ('SVR 3')
    reg = svm.SVR(C=1000, epsilon=0.1, kernel='rbf', degree=3)
    z = reg.fit(X_train, y_train).predict(X_test)
    print ("regression fit score: " + str(reg.score(X_train, y_train)))
    print ('R2: ' + str(sl.metrics.r2_score(y_test, z)))
    print ('mae: ' + str(mae(y_test, z)))
    # print ('conf matrix 2: ' + str(sl.metrics.confusion_matrix(y_test, z)))
    # print ('class report 2: ' + str(sl.metrics.classification_report(y_test, z)))

    print ('ExtraTreeRegressor')
    reg = sl.tree.ExtraTreeRegressor()
    z = reg.fit(X_train, y_train).predict(X_test)
    print ("regression fit score: " + str(reg.score(X_train, y_train)))
    print ('R2: ' + str(sl.metrics.r2_score(y_test, z)))
    print ('mae: ' + str(mae(y_test, z)))
    # print ('conf matrix 2: ' + str(sl.metrics.confusion_matrix(y_test, z)))
    # print ('class report 2: ' + str(sl.metrics.classification_report(y_test, z)))

    print ('LogisticRegression')
    reg = linear_model.LogisticRegression()
    z = reg.fit(X_train, y_train).predict(X_test)
    print ("regression fit score: " + str(reg.score(X_train, y_train)))
    print ('R2: ' + str(sl.metrics.r2_score(y_test, z)))
    print ('mae: ' + str(mae(y_test, z)))
    # print ('conf matrix 2: ' + str(sl.metrics.confusion_matrix(y_test, z)))
    # print ('class report 2: ' + str(sl.metrics.classification_report(y_test, z)))

    print ('KNeighborsClassifier')
    reg = neighbors.KNeighborsClassifier(3, weights='distance')
    z = reg.fit(X_train, y_train).predict(X_test)
    print ("regression fit score: " + str(reg.score(X_train, y_train)))
    print ('R2: ' + str(sl.metrics.r2_score(y_test, z)))
    print ('mae: ' + str(mae(y_test, z)))
    # print ('conf matrix 2: ' + str(sl.metrics.confusion_matrix(y_test, z)))
    # print ('class report 2: ' + str(sl.metrics.classification_report(y_test, z)))


if __name__=='__main__':

    args = { 'in_dir':  '/Users/dan/dev/datasci/kaggle/loan/',
             'out_dir': '/Users/dan/dev/datasci/kaggle/loan/'}
    model = main(**args)
