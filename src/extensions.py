# -*- coding: utf-8 -*-

# utils.py
# load and split data about states
def load_ACSPublicCoverage(subset, states=states, source_year="2017", target_year="2017", task_method=ft.ACSPublicCoverage):
    # Dictionaries mapping states to train-test data
    X_train_s, X_test_s, y_train_s, y_test_s = dict(), dict(), dict(), dict()
    for s in states:
        print(s, end=' ')
        source_data = load_folktables_data(s, source_year, '1-Year', 'person')  
        features_s, labels_s, group_s = task_method.df_to_numpy(source_data)
        X_s = pd.DataFrame(features_s, columns=task_method.features)
        X_s['y'] = labels_s
        y_s = X_s['y']
        X_s = X_s[subset]
        X_train_s[s], X_test_s[s], y_train_s[s], y_test_s[s] = split_data(X_s, y_s)
        
    if target_year == source_year:
        X_train_t, X_test_t, y_train_t, y_test_t = X_train_s, X_test_s, y_train_s, y_test_s
    else:
        X_train_t, X_test_t, y_train_t, y_test_t = dict(), dict(), dict(), dict()
        for s in states:
            print(s, end=' ')
            target_data = load_folktables_data([s], target_year, '1-Year', 'person')  
            features_t, labels_t, group_t = task_method.df_to_numpy(target_data)
            X_t = pd.DataFrame(features_t, columns=task_method.features)
            X_t['y'] = labels_t
            y_t = X_t['y']
            X_t = X_t[subset]
            X_train_t[s], X_test_t[s], y_train_t[s], y_test_t[s] = split_data(X_t, y_t)
    return X_train_s, X_test_s, y_train_s, y_test_s, X_train_t, X_test_t, y_train_t, y_test_t 
