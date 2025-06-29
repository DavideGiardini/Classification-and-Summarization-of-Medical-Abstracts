from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tqdm import tqdm


def build_results(FE_train, FE_test, train, test):
    results = test.copy()

    for label in range(1,6):
        label = str(label)
        y_test = list(test[label])

        clf = DecisionTreeClassifier().fit(FE_train, list(train[label]))
        y_pred = clf.predict(FE_test)
        new_col = str(label) + ' DT'
        results[new_col] = y_pred
        #print("DT done")

        clf = RandomForestClassifier().fit(FE_train, list(train[label]))
        y_pred = clf.predict(FE_test)
        new_col = str(label) + ' RF'
        results[new_col] = y_pred
        #print("RF done")

        clf = GaussianNB().fit(FE_train, list(train[label]))
        y_pred = clf.predict(FE_test)
        new_col = str(label) + ' NB'
        results[new_col] = y_pred
        #print("NB done")
        
    #results = results[['medical_abstract', '1', '1 DT', '1 RF', '1 NB', '2', '2 DT', '2 RF', '2 NB', '3', '3 DT', '3 RF', '3 NB', '4', '4 DT', '4 RF', '4 NB', '5', '5 DT', '5 RF', '5 NB']]

    
    return results


def build_resultsSVM(FE_train, FE_test, train, test):
    results = test.copy()

    for label in range(1,6):
        label = str(label)
        y_test = list(test[label])

        clf = DecisionTreeClassifier().fit(FE_train, list(train[label]))
        y_pred = clf.predict(FE_test)
        new_col = str(label) + ' DT'
        results[new_col] = y_pred
        #print("DT done")

        clf = RandomForestClassifier().fit(FE_train, list(train[label]))
        y_pred = clf.predict(FE_test)
        new_col = str(label) + ' RF'
        results[new_col] = y_pred
        #print("RF done")

        clf = GaussianNB().fit(FE_train, list(train[label]))
        y_pred = clf.predict(FE_test)
        new_col = str(label) + ' NB'
        results[new_col] = y_pred
        #print("NB done")
        
        clf = SVC().fit(FE_train, list(train[label]))
        y_pred = clf.predict(FE_test)
        new_col = str(label) + ' SVM'
        results[new_col] = y_pred
        #print("NB done")
    
    return results


def macro_f1(results):
    f1_score_diz = {}
    
    for classifier in ['DT', 'RF', 'NB']:
        lst = []
        for label in range(1,6):
            label = str(label)
            lst.append(f1_score(results[label], results[str(label) + ' ' + classifier]))
        f1_score_diz[classifier] = round(sum(lst) / len(lst), 3)
        
    return f1_score_diz


def micro_f1(results):
    f1_score_diz = {}
    
    for classifier in ['DT', 'RF', 'NB']:
        TP_list = []
        FP_list = []
        FN_list = []

        for label in range(1,6):
            label = str(label)
            # TP = Number of times in which both column "1" and "1 DT" are = 1
            TP = len(results[(results[label] == 1) & (results[str(label) + ' ' + classifier] == 1)])
            TP_list.append(TP)
            # FP = Number of times in which column "1" = 0 and column "1 DT" = 1
            FP = len(results[(results[label] == 0) & (results[str(label) + ' ' + classifier] == 1)])
            FP_list.append(FP)
            # FN = Number of times in which column "1" = 1 and column "1 DT" = 0
            FN = len(results[(results[label] == 1) & (results[str(label) + ' ' + classifier] == 0)])
            FN_list.append(FN)

        # Compute denominators
        den_precision = [x + y for x, y in zip(TP_list, FP_list)]
        den_recall = [x + y for x, y in zip(TP_list, FN_list)]

        # Compute precision and recall
        precision = sum(TP_list) / sum(den_precision)
        recall = sum(TP_list) / sum(den_recall)

        # Compute f1_score
        f1_score_diz[classifier] = round(2*precision*recall / (precision + recall), 3)
        
    return f1_score_diz


def micro_f1SVM(results):
    f1_score_diz = {}
    
    for classifier in ['DT', 'RF', 'NB', 'SVM']:
        TP_list = []
        FP_list = []
        FN_list = []

        for label in range(1,6):
            label = str(label)
            # TP = Number of times in which both column "1" and "1 DT" are = 1
            TP = len(results[(results[label] == 1) & (results[str(label) + ' ' + classifier] == 1)])
            TP_list.append(TP)
            # FP = Number of times in which column "1" = 0 and column "1 DT" = 1
            FP = len(results[(results[label] == 0) & (results[str(label) + ' ' + classifier] == 1)])
            FP_list.append(FP)
            # FN = Number of times in which column "1" = 1 and column "1 DT" = 0
            FN = len(results[(results[label] == 1) & (results[str(label) + ' ' + classifier] == 0)])
            FN_list.append(FN)

        # Compute denominators
        den_precision = [x + y for x, y in zip(TP_list, FP_list)]
        den_recall = [x + y for x, y in zip(TP_list, FN_list)]

        # Compute precision and recall
        precision = sum(TP_list) / sum(den_precision)
        recall = sum(TP_list) / sum(den_recall)

        # Compute f1_score
        f1_score_diz[classifier] = round(2*precision*recall / (precision + recall), 3)
        
    return f1_score_diz