import time
import pandas as pd
import numpy as np
import os

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler

from pyod.utils.data import evaluate_print
from pyod.models.abod import ABOD
from pyod.models.copod import COPOD
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.lscp import LSCP
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.mcd import MCD
from pyod.models.hbos import HBOS
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL

import settings as stt
from utils import calculate_eer, compute_AUC_EER
from feature_extraction import calculate_features


def df_to_array(df, classid=True):
    array = df.values
    _, cols = array.shape
    if classid:
        return array[:, 0 : cols - 1]
    else:
        return array[:, 0:cols]


def main():
    ROC_DIR = 'output_roc_data'
    results_filename = 'results_' + str(stt.NUM_ACTIONS) + '.txt'
    f_results = open(results_filename, 'w')
    try:
        os.mkdir(ROC_DIR)
    except OSError:
        print('Directory %s already exist' % ROC_DIR)
    else:
        print('Successfuly created the directory %s' % ROC_DIR)
    # BEGIN
    directory = "output_scores"
    path = os.path.join(".", directory)
    mode = 0o666
    try:
        os.mkdir(path, mode)
    except:
        print(directory + " already exists")
    # END 
    # --- Change here: Replace the synthetic filenames list with your own human data file(s) ---
    human_comparison_filenames = [
        "ownhuman_actions/actions_3min_dx_dy.csv"
    ]
    # --- Optionally update the corresponding labels ---
    comparison_labels = [
        'Your_Human_Data'
    ]
    
    roc_filelist = [
        ROC_DIR + '/own_human_data.csv',
    ]

    labels = [
        'bachelorarbeit_human',
    ]

    #  training data
    df_human_train = pd.read_csv("sapimouse_actions/actions_3min_dx_dy.csv", header=None)
    df_human_test = pd.read_csv("sapimouse_actions/actions_1min_dx_dy.csv", header=None)
   
    df_human_train = calculate_features(df_human_train)
    human_train = df_to_array(df_human_train, classid=False)
    scaler = MinMaxScaler()
    scaler.fit(human_train)
    human_train = scaler.transform(human_train) 
    # human test data
    df_human_test = calculate_features(df_human_test)
    human_test = df_to_array(df_human_test, classid=False)
    human_test = scaler.transform(human_test)
        
    models = []
    # Uncomment or add models as needed:
    # models.append(("PCA", PCA(random_state=stt.RANDOM_STATE)))
    # models.append(("MCD", MCD()))
    # models.append(("OCSVM", OCSVM()))
    # models.append(("LOF", LOF()))
    # models.append(("CBLOF", CBLOF()))    
    # models.append(("HBOS", HBOS()))
    # models.append(("KNN", KNN()))
    # models.append(("ABOD", ABOD()))
    # models.append(("COPOD", COPOD()))
    # models.append(("IForest", IForest(random_state=stt.RANDOM_STATE)))  
    models.append(("FeatureBagging", FeatureBagging(random_state=stt.RANDOM_STATE))) 
    # models.append(("LSCP", LSCP([LOF(), LOF()], random_state=stt.RANDOM_STATE)))  
    # models.append(("SO_GAAL", SO_GAAL()))
    # models.append(("MO_GAAL", MO_GAAL()))
    name_list = []
    # decision based on num_scores samples
    num_scores = stt.NUM_ACTIONS
    # for each model
    for name, model in models:
        print(name)
        name_list.append(name)
        # train the model with human data
        clf = model
        clf.fit(human_train)
        # evaluate the model with human data
        positive_scores = clf.decision_function(human_test)
        ps = list()
        for i in range(0, len(positive_scores) - num_scores + 1):
            sum_scores = 0
            for j in range(i, i + num_scores):
                sum_scores = sum_scores + positive_scores[j]
            ps.append(sum_scores / num_scores)
        positive_scores = np.array(ps)
        
        auc_list = []
        eer_list = []
        # --- Change here: Instead of synthetic test data, loop over your own human data file(s) ---
        index = 0
        for filename in human_comparison_filenames:
            df_comparison = pd.read_csv(filename, header=None)
            df_comparison = calculate_features(df_comparison)
            comparison_test = df_to_array(df_comparison, classid=False)
            comparison_test = scaler.transform(comparison_test)
            
            # evaluate the model with your human comparison data
            negative_scores = clf.decision_function(comparison_test)
            ps = list()
            for i in range(0, len(negative_scores) - num_scores + 1):
                sum_scores = 0
                for j in range(i, i + num_scores):
                    sum_scores = sum_scores + negative_scores[j]
                ps.append(sum_scores / num_scores)
            negative_scores = np.array(ps)
            
            # 0 - inlier; 1 - outlier
            zeros = np.zeros(len(positive_scores))
            ones = np.ones(len(negative_scores))
            y = np.concatenate((zeros, ones), axis=0)
            y_pred = np.concatenate((positive_scores, negative_scores), axis=0)
            
            # Save scores
            output = pd.DataFrame({'label': y, 'score': y_pred})
            # If you have only one comparison file, you may want to change the output filename accordingly:
            output.to_csv('output_scores/' + name + '_' + str(num_scores) + '_' + comparison_labels[index] + '.csv', index=False)
            
            evaluate_print(clf, y, y_pred)
            auc = np.round(roc_auc_score(y, y_pred), decimals=4)
            fpr, tpr, thr = roc_curve(y, y_pred, pos_label=1)
            eer = calculate_eer(y, y_pred)
            
            print(auc)
            print(eer)
            
            eer_list.append(eer)
            auc_list.append(auc)
            # Save ROC data if needed
            roc_data = True
            if roc_data:
                roc_dict = {"FPR": fpr, "TPR": tpr}
                df_roc = pd.DataFrame(roc_dict)
                # Save file with a name based on the index if needed; you might update roc_filelist accordingly.
                df_roc.to_csv(ROC_DIR + '/comparison_' + str(index) + '.csv', index=False)
            index = index + 1
        result = name + '&'
        for idx in range(len(comparison_labels)):
            result = result + '{auc:5.2f}& {eer:5.2f}&'.format(auc=auc_list[idx], eer=eer_list[idx])
        print(result)
        f_results.write(result+'\n')
        for idx in range(len(comparison_labels)):
            print('{0:}, {1:5.2f}, {2:5.2f}'.format(comparison_labels[idx], auc_list[idx], eer_list[idx]))   
    f_results.close()
    

roc_data = True
if __name__ == "__main__":
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f"Execution time: {toc - tic:0.4f} seconds")