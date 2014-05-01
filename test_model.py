__author__ = 'dan'

'''
Utility file to score predictions
'''

import pandas as pd
import sklearn as sl
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier


#use_pca=True
use_pca=False

#dir_base = 'full'
dir_base = 'short'
out_inter = '/int'
out_final = '/fin'


def loaddata(in_file):
    print "reading datafile: " + str(in_file)
    df = pd.read_table(in_file, sep=',')
    return df

def dwl(val):
    if val > 0:
        return 1
    else:
        return 0


def main(in_dir, out_dir):

    train_file = out_dir + out_inter + '/train_v2_pca.csv'
    if use_pca == False:
        train_file = out_dir + out_inter + '/train_v2_clean_scaled.csv'

    dftrain = loaddata(train_file)
    print("full training set size=" + str(dftrain.shape))

    # add classification column
    dftrain['dwl'] = dftrain['loss'].apply(lambda x: dwl(x) )

    # split out regression set
    dftrain_reg = dftrain[dftrain['dwl'] == 1]
    excludes = ['id','f776','f777','f778','loss', 'dwl']
    X2 = dftrain_reg.drop(excludes, axis=1)
    y2 = dftrain_reg['loss']

    X = dftrain.drop(excludes, axis=1)
    y = dftrain['dwl']
    print("classifier training set size=" + str(X.shape) + " | num dwl: " + str(y.sum()) )

    ############
    ## slim down features to most important
    ############
    clf = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                                          C=1.0, fit_intercept=True, intercept_scaling=1.0,
                                          class_weight=None, random_state=None)
    # clf = sl.ensemble.ExtraTreesClassifier()
    # clf = linear_model.RidgeClassifier()
    # clf = RandomForestClassifier()
    X_new = clf.fit(X, y).transform(X)
    print(str(X_new.shape))
    # importances = clf.feature_importances_
    # print(str(np.sort(importances)) + " | sum=" + str(np.sum(importances)))

    ############
    ## first classification
    ############
    print("starting classification x-val")
    clf.fit(X_new, y)
    # scores = cross_validation.cross_val_score(clf, X_new, y, scoring='recall')
    # print ("classifier xval: " + str(scores))
    y_pred = clf.predict(X_new)

    dftrain['pred'] = y_pred
    recall_set = dftrain[['dwl','pred']][dftrain['dwl'] == 1]
    print ("my recall calc: " + str(1.0*recall_set['pred'].sum() / recall_set['dwl'].sum()))
    print ('conf matrix: ' + str(sl.metrics.confusion_matrix(y, y_pred)))
    print ('class report: ' + str(sl.metrics.classification_report(y, y_pred)))

    ################
    ## simple cross validation
    ################
    a_train, a_test, b_train, b_test = sl.cross_validation.train_test_split(X_new, y)
    z = clf.fit(a_train, b_train).predict(a_test)
    print ('conf matrix2: ' + str(sl.metrics.confusion_matrix(b_test, z)))
    print ('class report2: ' + str(sl.metrics.classification_report(b_test, z)))


    ################
    ## run regression
    ################
    print("regression training set size=" + str(X.shape))
    #reg = sl.linear_model.LinearRegression()
    # reg = svm.SVR()
    # reg = gaussian_process.GaussianProcess(regr='quadratic')
    reg = sl.tree.ExtraTreeRegressor()
    print("starting regression x-val")
    reg.fit(X2, y2)
    X2_new = reg.transform(X2)
    reg.fit(X2_new, y2)
    # scores = cross_validation.cross_val_score(reg, X2, y2, scoring='mean_squared_error')
    scores = cross_validation.cross_val_score(reg, X2_new, y2, scoring='mean_squared_error')
    print ("regression xval: " + str(scores))

    # y2_pred = reg.predict(X2)
    y2_pred = reg.predict(X2_new)
    dftrain_reg['y2_pred'] = y2_pred
    print ("loss values: " + str(dftrain_reg[['loss','y2_pred']].head(50)))



    ################
    ## simple cross validation
    ################
    # # a_train, a_test, b_train, b_test = sl.cross_validation.train_test_split(X2, y2)
    # a_train, a_test, b_train, b_test = sl.cross_validation.train_test_split(X2_new, y2)
    # z = reg.fit(a_train, b_train).predict(a_test)
    # print ('R2: ' + str(sl.metrics.r2_score(b_test, z)))
    # print (str(b_test[1:25]) + " | " + str(z[1:25]))





if __name__=='__main__':

    args = { 'in_dir':  '/Users/dan/dev/datasci/kaggle/loan/' + dir_base,
             'out_dir': '/Users/dan/dev/datasci/kaggle/loan/' + dir_base}
    model = main(**args)

#    print "args=" + str(sys.argv[0:])
#    model = main(sys.argv[1], sys.argv[2])
