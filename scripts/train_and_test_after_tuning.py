from __future__ import division
import numpy as np
# import matplotlib.pyplot as plt
from time import time
# matplotlib inline
from sklearn.svm import SVC
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#Data = np.loadtxt('Olaa_33' + '.csv', delimiter=',')

#csvfile = pd.read_csv('Labeled_data_variant_fea_added.csv', header=0)
#csvfile2 = pd.read_csv('Test_difference_data3.csv', header=0)
#clin_var = pd.read_csv('ClinVar.Reviewed.2019-11_data33.csv', header=0)
#csvfile3 = pd.concat([csvfile, csvfile2, clin_var], ignore_index=True)
#csvfile4 = pd.read_csv('all_data.without_the_32_pathogenic.fst.2.csv', header=0)

#csvfile5 = pd.merge(csvfile3, csvfile4,on=["variant", "variant"])
#csvfile3 = csvfile5
#csvfile3.to_csv('data_with_fst.csv', sep=',', na_rep='nan', header=True, index=False)

csvfile3 = pd.read_csv('data_with_fst_2.csv', header=None)

#Repeats = 30
Repeats = 1

#Cs = [1, 10, 100, 1000]
Cs = [100]

f1_valid = np.zeros(len(Cs))
recall_valid = np.zeros(len(Cs))
precision_valid = np.zeros(len(Cs))
acc_valid = np.zeros(len(Cs))
f1_test = np.zeros(len(Cs))
recall_test = np.zeros(len(Cs))
precision_test = np.zeros(len(Cs))
acc_test = np.zeros(len(Cs))

f2 = open("CV5_test_40.txt", "w")

for k in range(Repeats):
    #train, test = train_test_split(csvfile3, test_size=0.2, random_state=k)
    train, test = train_test_split(csvfile3, test_size=0.3, random_state=k)
    train.to_csv('training_data.csv', sep=',', na_rep='nan', header=False, index=False)
    test.to_csv('testing_data.csv', sep=',', na_rep='nan', header=False, index=False)
    # train = pd.concat([train, test], ignore_index=False)
    #csvfile2= train.loc[:, train.columns != 44]
    csvfile2= train
    #variants_col= train.loc[:, train.columns == 0]
    Data = csvfile2.to_numpy()
    #csvfile2= test.loc[:, test.columns != 44]
    csvfile2= test
    Data_test = csvfile2.to_numpy()

    #Data = np.loadtxt('Labeled_data_variant_fea_added' + '.csv', delimiter=',')
    #BRCAEx.prepared.manually.unseen_data3.csv
    # Data_no_label = np.loadtxt('Olaa_2_updated_no_label' + '.csv', delimiter=',')
    #Data_no_label = np.loadtxt('ClinVar.Reviewed.2019-11_data3' + '.csv', delimiter=',')
    Data_no_label = np.loadtxt('./WES_real/DoCM.indels.unseen.2' + '.csv', delimiter=',')
    # Data_no_label = np.loadtxt('WES_simulated/TSVC_variants.vcf.File.No.939503.annotation.hg19_multianno.prepared.2_data3' + '.csv', delimiter=',')
    #Data_no_label = np.loadtxt('WES_simulated/TSVC_variants.vcf.File.No.5060137.annotation.hg19_multianno.prepared.2_data3' + '.csv', delimiter=',')
    #Data_no_label = np.loadtxt('WES_real/19DG0041.annotation.hg19_multianno.prepared.2_data3' + '.csv', delimiter=',')
    #Data_no_label = np.loadtxt('WES_real/19DG0132.annotation.hg19_multianno.prepared.2_data3' + '.csv', delimiter=',')

    # Data_extra_pathogenic = np.loadtxt('Test_difference_data3' + '.csv', delimiter=',')

    print("data has been read")
    print(Data.shape)
    X = Data[:, :-1]
    X_test = Data_test[:, :-1]
    X = X[:, 4:]
    X_test = X_test[:, 4:]
    print(X.shape)
    print(X[0,0])
    #X = np.delete(X, 2, 1)
    #X = np.delete(X, 1, 1)
    #X = np.delete(X, 0, 1)
    #X = np.delete(X, 1, 1)
    #X = np.delete(X, 0, 1)

    y = Data[:, -1]
    y_test_f = Data_test[:, -1]
    
    # without fst
    X = X[:, :-1]
    X_test = X_test[:, :-1]
    X = np.nan_to_num(X)
    X_test = np.nan_to_num(X_test)
    X1 = Data_no_label[:, :-1]
    X1 = X1[:, 4:]
    X1 = np.nan_to_num(X1)
    len_x = X.shape[0]
    print(X1.shape)

 # to remove the Allel Frequency (AF) features.
    #list_unique = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,40]
    #list_unique = [1, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    #X = np.delete(X, list_unique, 1)
    #print(list_unique)
    #X_test = np.delete(X_test, list_unique, 1)
    #X1 = np.delete(X1, list_unique, 1)


    corrs = np.corrcoef(X, y, rowvar=False)
    len1 = corrs.shape[0]
    corr_y = corrs[:, len1-1]
    corr_y1 = corrs[:len1-1, :len1-1]
#    SGPFreq = corrs[:, 28]
#    SGP_AF_Strict = corrs[:, 29]
    np.savetxt("corr_y1.csv", corr_y1, delimiter=",")

    mut_info = mutual_info_classif(X, y)
    removed_indices = []
    for i in range(0, corr_y1.shape[0]):
        for j in range(i+1, corr_y1.shape[1]):
            if corr_y1[i, j] > 0.98:
                removed_indices.append(j)
    list_unique = list(set(removed_indices))
    X = np.delete(X, list_unique, 1)
    print(list_unique)
    X_test = np.delete(X_test, list_unique, 1)
    X1 = np.delete(X1, list_unique, 1)

    #X_test_f = np.delete(X_test_f, list_unique, 1)
    print(X.shape)
    mi = mut_info.reshape(mut_info.shape[0], 1)
    corr_y = corr_y.reshape(corr_y.shape[0], 1)
#    SGP_AF_Strict = SGP_AF_Strict.reshape(SGP_AF_Strict.shape[0], 1)
#    SGPFreq = SGPFreq.reshape(SGPFreq.shape[0], 1)
#    new2 = np.concatenate((SGP_AF_Strict, corr_y), axis=1)
#    new1 = np.concatenate((SGPFreq, new2), axis=1)
#
#    np.savetxt('mutual_info1.csv', new1, delimiter=',', fmt='%.2f')
    scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    X = scaler.fit_transform(X)
    X1 = scaler.transform(X1)
    X_test = scaler.transform(X_test)

    #X = SelectKBest(chi2, k=20).fit_transform(X, y)
    selector = SelectKBest(chi2, k=20)
    selector.fit(X, y)
    X = selector.transform(X)
    X1 = selector.transform(X1)
    X_test = selector.transform(X_test)
    cols = selector.get_support(indices=True)
    print(cols)
    #cols = [0,1,2,3,4,5,6,7,8,9,10,19]
    #cols = [11,12,13,14,15,16,17,18]
    #print(X[1,19])
    #X = X[:, :-1]
    #X1 = X1[:, :-1]
    #X_test = X_test[:, :-1]
    
    #X = np.concatenate((X, X1), axis=0)
    scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    #X1 = scaler.transform(X1)
    #X_test_f = scaler.transform(X_test)
    X_test_f = X_test
    print(X.shape)
    print(X1.shape)
    print(X_test_f.shape)
    print("k = %d" % (k))
    # Gammas = ['auto', 'scale']
    Gammas = ['rbf']

    # svclassifier = RandomForestClassifier(n_estimators=100, random_state=0)
    c = Cs[0] # the best value after tuning
    #svclassifier = RandomForestClassifier(n_estimators=c, random_state=0, class_weight={0:2,1:10})
    svclassifier = RandomForestClassifier(n_estimators=c, random_state=0)
    total_prec_t = 0.0
    total_recall_t = 0.0
    total_f1_score_t = 0.0
    total_Accuracy_t = 0.0

    counter1 = 0
    X_train = X
    y_train = y
    y_train = y_train.ravel()

    t0 = time()
    svclassifier.fit(X_train, y_train)
    print("Linear Kernel Normalized Fit Time: %.4f s" % (time()-t0))
    y_pred_t = svclassifier.predict(X_test_f)
    y_pred_x1 = svclassifier.predict(X1)
    with open('predicting_with_label.csv', 'w') as f10:
        for items in y_pred_t:
            #for item in items:
            f10.write("%d\n" % int(items))
            #f10.write("%s" % str(np.around(items, decimals=0)))
    # WES_simulated/442896
    # WES_simulated/939503
    # WES_simulated/5060137
    # WES_real/19DG0041
    # WES_real/19DG0132
    with open('./WES_real/DoCM.indels.unseen.2_20_feas.csv', 'w') as f10:
        for items in y_pred_x1:
            #for item in items:
            f10.write("%d\n" % int(items))
            #f10.write("%s" % str(np.around(items, decimals=0)))

    conf_mat_t = confusion_matrix(y_test_f, y_pred_t)
    tn, fp, fn, tp = confusion_matrix(y_test_f, y_pred_t).ravel()
    tp_t = conf_mat_t[0, 0]
    fn_t = conf_mat_t[0, 1]
    fp_t = conf_mat_t[1, 0]
    tn_t = conf_mat_t[1, 1]
    print("tn= %d, fp= %d, fn= %d, tp= %d" % (tn_t, fp_t, fn_t, tp_t))
    print("tn= %d, fp= %d, fn= %d, tp= %d" % (tn, fp, fn, tp))
    recall_class1_t = tp_t / (tp_t+fn_t) # sensitivity
    recall_class2_t = tn_t / (tn_t+fp_t) # specificity
    precision_class1_t = tp_t / (tp_t+fp_t)
    precision_class2_t = tn_t / (tn_t+fn_t)

    total_prec_t =total_prec_t+precision_class2_t
    total_recall_t = total_recall_t + recall_class2_t

    precision_class2_t = tn_t / (tn_t+fn_t)
    Accuracy_t = (tp_t+tn_t)/ (tp_t+tn_t+fn_t+fp_t)
    total_Accuracy_t = total_Accuracy_t + Accuracy_t

    f1_score_t = 2 * ((precision_class2_t * recall_class2_t) / (precision_class2_t + recall_class2_t))
    total_f1_score_t = total_f1_score_t + f1_score_t

    final_prec_c1 = total_prec_t
    final_rec_c1 = total_recall_t
    final_f1 = total_f1_score_t
    final_acc = total_Accuracy_t

    f2.write("%d, %d, %.4f, %.4f, %.4f, %.4f \n" % (k,c, final_prec_c1, final_rec_c1, final_f1, final_acc))

f2.close()
