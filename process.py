__author__ = 'dan'


import os
import sys
import xml.dom.minidom as xml
import pandas as pd
import numpy as np
import random
from pandas.core.series import Series
from pandas.tseries.index import date_range
import matplotlib as mp
import matplotlib.pyplot as plt
import pylab
import datetime
import csv
import sklearn as sl
from sklearn import neighbors
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score

#steps = [1,2,3,4,5] # all
#steps = [3,4,5] # pca+
steps = [4,5] # train+
#steps = [5] # test+
#steps = [4] # train only
#steps = [3,4] # pca and train only

#use_pca=True
use_pca=False

dir_base = 'full'
#dir_base = 'short'
out_inter = '/int'
out_final = '/fin'

def cat_and_clean(root, dirs, out_dir):
    print ("done with cat and clean")

def strip_junk(df):
    #print y
    return df

def timedelta(x):
    i = None
    if (x[0] > x[1]):
        i = x[0] - x[1]
    else:
        i = x[1] - x[0]
    secs = i.total_seconds()
    return secs*1000

def mae(y_pred, y_act):
    return (np.abs(y_act - y_pred).sum() * 1.0)/len(y_pred)

def dwl(val):
    if val > 0:
        return 1
    else:
        return 0


def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]


def clean(df):
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    dfc = pd.DataFrame(data=imp.fit_transform(df), columns=df.columns)
    for col in dfc.columns:
        desc = dfc[col].describe()
        if desc['std'] == 0:
            print('dropping column: ' + col)
            dfc = dfc.drop(col, axis=1)
    return dfc


def loaddata(in_file):
    print "reading datafile: " + str(in_file)
    df = pd.read_table(in_file, sep=',')
    return df


def clean_and_save(in_dir, out_dir):
    train_file = in_dir + '/train_v2.csv'
    dftrain = loaddata(train_file)
    dftrain = clean(dftrain)
    dftrain.to_csv(out_dir + out_inter + '/train_v2_clean.csv', index=False)

    test_file = in_dir + '/test_v2.csv'
    dftest = loaddata(test_file)
    dftest = clean(dftest)
    dftest.to_csv(out_dir + out_inter + '/test_v2_clean.csv', index=False)


def scale_and_save(in_dir, out_dir):
    excludes = ['id','f776','f777','f778','loss']

    sca = preprocessing.StandardScaler()

    train_file = in_dir + '/train_v2_clean.csv'
    dftrain = loaddata(train_file)
    print("orig train cols: " + str(len(dftrain.columns)))
    dftrainD = dftrain.drop(excludes, axis=1)
    print("train cols to scale: " + str(len(dftrainD.columns)))
    sca.fit(dftrainD)
    print("done training scale model")
    dftrain[dftrainD.columns] = sca.transform(dftrainD)
    print("done scalng training set")
    dftrain.to_csv(out_dir + out_inter + '/train_v2_clean_scaled.csv', index=False)

    excludes = ['id','f776','f777','f778']

    test_file = in_dir + '/test_v2_clean.csv'
    dftest = loaddata(test_file)
    print("orig test cols: " + str(len(dftest.columns)))
    dftestD = dftest.drop(excludes, axis=1)
    print("test cols to scale: " + str(len(dftestD.columns)))
    dftest[dftestD.columns] = sca.transform(dftestD)
    print("done scalng test set")
    dftest.to_csv(out_dir + out_inter + '/test_v2_clean_scaled.csv', index=False)


def reduce(in_dir, out_dir):
    excludes = ['id','f776','f777','f778','loss']

    # pca = sl.decomposition.PCA(n_components = 'mle')
    # pca = sl.tree.ExtraTreeRegressor()
    pca = LinearSVC(C=0.01, penalty="l1", dual=False, verbose = 2)

    train_file = in_dir + '/train_v2_clean_scaled.csv'
    dftrain = loaddata(train_file)
    dftrainD = dftrain.drop(excludes, axis=1)
    print("train cols to reduce: " + str(len(dftrainD.columns)))
    dftrainS = dftrain[excludes]
    pca.fit(dftrainD, dftrain['loss'])
    print("done training reduce model.")
    pcad = pd.DataFrame(pca.transform(dftrainD))
    dftrain = dftrainS
    dftrain[pcad.columns] = pcad
    print("done transforming training set; new size=" + str(dftrain.shape))
    dftrain.to_csv(out_dir + out_inter + '/train_v2_pca.csv', index=False)

    excludes = ['id','f776','f777','f778']

    test_file = in_dir + '/test_v2_clean_scaled.csv'
    dftest = loaddata(test_file)
    dftestD = dftest.drop(excludes, axis=1)
    print("test cols to reduce: " + str(len(dftrainD.columns)))
    dftestS = dftest[excludes]
    pcad = pd.DataFrame(pca.transform(dftestD))
    dftest = dftestS
    dftest[pcad.columns] = pcad
    print("done transforming test set; new size=" + str(dftest.shape))
    dftest.to_csv(out_dir + out_inter + '/test_v2_pca.csv', index=False)


def experiment(dfX_train, dfX_test, dfy_train, dfy_test):
    res = []

    for n_neighbors in [3,5,10,15,20]:
        for weights in ['uniform', 'distance']:
            print ('training KNeighborsClassifier model...' + str(n_neighbors) + '|' + weights)
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(dfX_train, dfy_train)
            pred = clf.predict(dfX_test)
            f1 = sl.metrics.f1_score(dfy_test, pred)
            recall = sl.metrics.recall_score(dfy_test, pred)
            print(str(recall))
            res.append(str(f1) + "|" + str(n_neighbors) + "|" + weights)
            print(str(f1))

    for n_estimators in [10,20,50,100,500]:
        for max_features in ['auto', 'sqrt', 'log2']:
            print ('training RandomForestClassifier model...' + str(n_estimators) + '|' + max_features)
            clf = sl.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, n_jobs=-1, verbose=0)
            clf.fit(dfX_train, dfy_train)
            pred = clf.predict(dfX_test)
            f1 = sl.metrics.f1_score(dfy_test, pred)
            recall = sl.metrics.recall_score(dfy_test, pred)
            print(str(recall))
            res.append(str(f1) + "|" + str(n_estimators) + "|" + max_features)
            print(str(f1))

    for C in [.1,.3,1,3,10]:
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
            print ('training SVC model...' + str(C) + '|' + kernel)
            clf = sl.svm.SVC(C=C, kernel=kernel, degree=3)
            clf.fit(dfX_train, dfy_train)
            pred = clf.predict(dfX_test)
            f1 = sl.metrics.f1_score(dfy_test, pred)
            recall = sl.metrics.recall_score(dfy_test, pred)
            print(str(recall))
            res.append(str(f1) + "|" + str(C) + "|" + kernel)
            print(str(f1))

    # for learning_rate in [.001,.003,.01,.03,.1,.3,1]:
    #     for n_components in [10,50,100,200,500]:
    #         print ('training Bernuili model...' + str(learning_rate) + '|' + str(n_components))
    #         clf = sl.neural_network.BernoulliRBM(n_components=n_components, learning_rate=learning_rate, batch_size=10, n_iter=10, random_state=None)
    #         clf.fit(dfX_train, dfy_train)
    #         pred = clf.predict(dfX_test)
    #         f1 = sl.metrics.f1_score(dfy_test, pred)
    #         recall = sl.metrics.recall_score(dfy_test, pred)
    #         print(str(recall))
    #         res.append(str(f1) + "|" + str(learning_rate) + '|' + str(n_components))
    #         print(str(f1))


    for score in res:
        print(str(score))


def train(in_dir):

    train_file = in_dir + '/train_v2_pca.csv'
    if use_pca == False:
        train_file = in_dir + '/train_v2_clean_scaled.csv'

    stats = []

    dftrain = loaddata(train_file)
    print("full training set size=" + str(dftrain.shape))


    # reda = LinearSVC(C=0.1, penalty="l1", dual=False, verbose = 0)
    # for n in [50, 100, 150, 200, 300]:
    for n in [150]:
        # add classification column
        dftrain['dwl'] = dftrain['loss'].apply(lambda x: dwl(x) )

        # split out regression set
        dftrain_reg = dftrain[dftrain['dwl'] == 1]

        # train classifier
        excludes = ['id','f776','f777','f778','loss', 'dwl']
        X = dftrain.drop(excludes, axis=1)
        X = dftrain[['f527','f528']]
        # X = np.array([dftrain['f528'] - dftrain['f527']]).T
        y = dftrain['dwl']
        print('DIAG X shape=' + str(np.shape(X)))
        # print('DIAG X cols=' + str(X.columns))

        reda = sl.feature_selection.SelectKBest(sl.feature_selection.f_classif, n)
        # print ('reducing X to # cols:' + str(n))
        # X = reda.fit(X, y).transform(X)
        print("classifier training set size=" + str(X.shape))

        # for w in [2,4,6,8,10,12,14,16]:
        for w in [12]:

            weights = {0:1,1:w}
            print('weights=' + str(weights))
            # clf = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
            #                                       C=1e20, fit_intercept=True, intercept_scaling=1.0,
            #                                       class_weight=weights, random_state=None)
            clf = linear_model.LogisticRegression(penalty='l2', C=1e20, class_weight=weights)
            # clf = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
            #                                       C=C, fit_intercept=True, intercept_scaling=1.0,
            #                                       class_weight=None, random_state=None)
            # print('LogisticRegression C=' + str(C))
            # clf = neighbors.KNeighborsClassifier(3, weights='distance')
            #clf = RandomForestClassifier()
            print("starting classification x-val")
            scores = clf.fit(X, y).score(X,y)
            # scores = cross_validation.cross_val_score(clf, X, y, scoring='recall')
            print ("classifier fit score: " + str(scores))

            a_train, a_test, b_train, b_test = sl.cross_validation.train_test_split(X, y)
            print ('clf roc_auc: ' + str(roc_auc_score(b_test,clf.predict_proba(a_test)[:,1])))
            #experiment(a_train, a_test, b_train, b_test)
            z = clf.predict(a_test)
            print ('conf matrix:\n' + str(sl.metrics.confusion_matrix(b_test, z)))
            print ('class report:\n' + str(sl.metrics.classification_report(b_test, z)))


            #train regression
            # redb = LinearSVC(C=0.1, penalty="l1", dual=False, verbose = 0)
            # for m in [50, 100, 150, 200, 300]:
            for m in [50]:
                redb = sl.feature_selection.SelectKBest(sl.feature_selection.f_classif, m)
                print ('reducing X2 to # cols:' + str(m))
                X2 = dftrain_reg.drop(excludes, axis=1)
                y2 = dftrain_reg['loss']
                X2 = redb.fit(X2, y2).transform(X2)
                print("regression training set size=" + str(X2.shape))
                # reg = linear_model.LinearSVC()
                # reg = svm.SVR()
                # reg = sl.tree.ExtraTreeRegressor()
                # reg = linear_model.LogisticRegression()
                # reg = neighbors.KNeighborsClassifier(3, weights='distance')
                reg = svm.SVR(C=10, epsilon=0.1, kernel='rbf', degree=3)

                print("starting regression x-val")
                scores = reg.fit(X2, y2).score(X2,y2)
                # scores = cross_validation.cross_val_score(reg, X, y, scoring='mean_squared_error')
                print ("regression fit score: " + str(scores))
                a_train, a_test, b_train, b_test = sl.cross_validation.train_test_split(X2, y2)
                # print ('reg roc_auc: ' + str(roc_auc_score(b_test,reg.predict_proba(a_test)[:,1])))
                z = reg.predict(a_test)
                print ('R2: ' + str(sl.metrics.r2_score(b_test, z)))
                # print ('conf matrix 2: ' + str(sl.metrics.confusion_matrix(b_test, z)))
                # print ('class report 2: ' + str(sl.metrics.classification_report(b_test, z)))


                # test the model
                dft = dftrain.drop(excludes, axis=1)
                print('DIAG dft shape=' + str(np.shape(dft)))
                # print('DIAG dft cols=' + str(dft.columns))
                # dftr = reda.transform(dft)
                dftr = dft[['f527','f528']]
                # dftr = np.array([dft['f528'] - dft['f527']]).T
                dftrain['dwl'] = clf.predict(dftr)
                dft = dftrain[dftrain['dwl'] == 1]
                dft = dft.drop(excludes, axis=1)
                dftr = redb.transform(dft)
                dft['lossp'] = reg.predict(dftr)
                dftrain['lossp'] = np.zeros(len(dftrain))
                dftrain.update(dft)
                maev = mae(dftrain['lossp'], dftrain['loss'])
                print('#################################')
                print('mae: ' + str(maev))
                print('#################################')
                dftrain = dftrain.drop('lossp', axis=1)

                print (str(b_test[1:25]) + " | " + str(z[1:25]))

                stats.append('mae' + str(maev) + "|" + str(sl.metrics.r2_score(b_test, z)) + "|" + str(n) + "|" + str(m) + "|" + str(w))


    for s in stats:
        print(s)

    return clf, reg, reda, redb


def dwl_rev(dwl, loss):
    if dwl == 1:
        return loss
    else:
        return 0

def test(in_dir, clf, reg, reda, redb, out_dir):

    test_file = in_dir + '/test_v2_pca.csv'
    if use_pca == False:
        test_file = in_dir + '/test_v2_clean_scaled.csv'

    dftest = loaddata(test_file)
    print("full test set size=" + str(dftest.shape))

    X = dftest.drop(['id','f776','f777','f778'], axis=1 )
    # X = reda.transform(X)
    # X = np.array([dftest['f528'] - dftest['f527']]).T
    X = dftest[['f527','f528']]
    print("classifier test set size=" + str(X.shape))
    class_res = clf.predict(X)
    dftest['dwl'] = class_res
    dftest['loss'] = np.zeros(len(dftest))
    dftest_reg = dftest[dftest['dwl'] == 1]
    print("loan default count=" + str(dftest['dwl'].sum()))

    X2 = dftest_reg.drop(['id','f776','f777','f778', 'dwl','loss'], axis=1 )
    X2 = redb.transform(X2)
    print("regression test set size=" + str(X2.shape))
    dftest_reg['loss'] = reg.predict(X2)

    dftest.update(dftest_reg)

    dftest[['id','loss']] = dftest[['id','loss']].astype(int)
    # dftest.to_csv(out_dir + out_final + '/test_v2_res.csv', index=False)
    dftest[['id','loss']].to_csv(out_dir + out_final + '/test_v2_submit.csv', index=False)
    print ("result: " + str(dftest[['id','loss']].head()))




def main(include_steps, in_dir, out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_dir + out_inter):
        os.makedirs(out_dir + out_inter)
    if not os.path.exists(out_dir + out_final):
        os.makedirs(out_dir + out_final)

    inter_dir = out_dir + out_inter

    if 1 in include_steps:
        # create clean files
        clean_and_save(in_dir,out_dir)

    if 2 in include_steps:
        # normalize
        scale_and_save(inter_dir,out_dir)

    if 3 in include_steps:
        # run PCA (train and test)
        reduce(inter_dir,out_dir)

    if 4 in include_steps:
        # train classification model
        # train dwl regression model
        clf, reg, reda, redb = train(inter_dir)

    if 5 in include_steps:
        # run class and reg on test
        test(inter_dir, clf, reg, reda, redb, out_dir)

if __name__=='__main__':

    args = { 'include_steps': steps,
             'in_dir':  '/Users/dan/dev/datasci/kaggle/loan/' + dir_base,
             'out_dir': '/Users/dan/dev/datasci/kaggle/loan/' + dir_base}
    model = main(**args)

#    print "args=" + str(sys.argv[0:])
#    model = main(sys.argv[1], sys.argv[2])
