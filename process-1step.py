__author__ = 'dan'

'''
Simple version of the full process file with everything in one method.
'''

import os

import pandas as pd
import numpy as np
import sklearn as sl
from sklearn.preprocessing import Imputer
from sklearn import preprocessing as pre
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sklearn.linear_model as lm


def mae(y_pred, y_act):
    return (np.abs(y_act - y_pred).sum() * 1.0)/len(y_pred)



def main(in_dir, out_dir):

    # read in training file
    print('reading train file...')
    df = pd.read_csv(in_dir + '/train_v2.csv')

    # clean
    imputer = Imputer()
    imputer.fit(df.drop(['loss'],axis=1))
    clean = imputer.transform(df.drop(['loss'],axis=1))

    # scale
    scaler = pre.StandardScaler()
    scaled = scaler.fit_transform(clean)
    dfs = pd.DataFrame(scaled, columns=df.columns[0:-1])
    dfs['loss'] = df['loss'].values
    dfs['loss_flag'] = dfs['loss'].apply(lambda x: 1 if x>0 else 0) # calculate whether or not the row represents a default

    # reduce for loan-default classification
    X1 = dfs[['f527','f528']]
    y1 = dfs['loss_flag']
    X1_train,X1_test,y1_train,y1_test = train_test_split(X1, y1)


    # train and x-validate
    clf = LogisticRegression(C=1e20,penalty='l2',class_weight={0:1,1:12})
    clf.fit(X1_train,y1_train)
    z = clf.predict(X1_test)
    print ('roc auc: ' + str(roc_auc_score(y1_test,z)))
    print ('conf matrix:\n' + str(sl.metrics.confusion_matrix(y1_test,z)))
    print ('class report:\n' + str(sl.metrics.classification_report(y1_test,z)))

    # reduce for loss regression
    dfs_reg = dfs[dfs['loss_flag'] == 1]  # select only rows that defaulted
    X2 = dfs_reg.drop(['id','f776','f777','f778','loss','loss_flag'], axis=1) # get rid of known unnecessary columns
    y2 = dfs_reg['loss']
    redb = sl.feature_selection.SelectKBest(sl.feature_selection.f_classif, 50)
    X2 = redb.fit(X2, y2).transform(X2) # reduce to top 50 features

    # train and x-validate
    X2_train,X2_test,y2_train,y2_test = train_test_split(X2, y2)
    reg = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                                          C=1.0, fit_intercept=True, intercept_scaling=1.0,
                                          class_weight=None, random_state=None)

    # reg = svm.SVR(C=1, epsilon=0.1, kernel='rbf', degree=3)
    z = reg.fit(X2_train, y2_train).predict(X2_test)
    print ('R2: ' + str(sl.metrics.r2_score(y2_test, z)))
    print ('mae: ' + str(mae(y2_test, z)))


    #######################################
    # read test file
    print('reading test file...')
    dft = pd.read_csv(in_dir + '/test_v2.csv')
    print('test size=' + str(dft.shape()))

    # clean
    imputer = Imputer()
    imputer.fit(dft)
    clean = imputer.transform(dft)

    # scale
    scaled = scaler.transform(clean)
    dftest = pd.DataFrame(scaled, columns=dft.columns)

    # apply loan-default classification to test set
    X1 = dftest[['f527','f528']]
    dftest['loss'] = np.zeros(len(dft)) # fill in a loss column with all zeros
    dftest['defaults'] = clf.predict(X1) # add a column of 1s and 0s to the main data frame indicating default or not
    print("loan default count=" + str(dftest['defaults'].sum()))

    # apply loss regression to subset of test set
    dft_reg = dftest[dftest['defaults'] == 1] # select only rows that were determined to be defaults in the previous step

    X2 = dft_reg.drop(['id','f776','f777','f778', 'defaults','loss'], axis=1) # get rid of known unnecessary columns
    X2 = redb.transform(X2) # reduce columns using the same formula as for the train set
    print("regression test set size=" + str(X2.shape))
    dft_reg['loss'] = reg.predict(X2)

    dftest.update(dft_reg) # merge the predicted loss value from the subset of rows in the "default" set with the full set

    dft['loss'] = dftest['loss'] # copy the loss values back to the original dataframe to align with id column

    # write results
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_dir + "/fin"):
        os.makedirs(out_dir + "/fin")
    dft[['id','loss']] = dft[['id','loss']].astype(int)
    dft[['id','loss']].to_csv(out_dir + '/fin/test_v2_submit.csv', index=False)
    print ("result: " + str(dft[['id','loss']].head()))


if __name__=='__main__':

    args = { 'in_dir':  '/Users/dan/dev/datasci/kaggle/loan/full/',
             'out_dir': '/Users/dan/dev/datasci/kaggle/loan/full/'}
    model = main(**args)
