
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
def classification_pipeline(X_train_data, X_test_data, y_train_data, test_generator, 
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       do_probabilities = False, title=None):
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=-1, 
        scoring=scoring_fit,
        verbose=2
    )
    fitted_model = gs.fit(X_train_data, y_train_data)
    
    if do_probabilities:
      pred = fitted_model.predict_proba(X_test_data)
    else:
      pred = fitted_model.predict(X_test_data)
    print(f'===================================== {title} =====================================')
    print(f'Best score: {fitted_model.best_score_}')
    print(f'Best params: {fitted_model.best_params_}')
    print(pred)
    y_pred = pred
    y_true = test_generator.classes
    y_true = y_true[0:len(y_pred)]
    ############## Confusion matrix ###############
    cnf_matrix = confusion_matrix(y_true, y_pred)
    print('Confusion matrix:')
    print(cnf_matrix)
    ############## Accuracy ###############
    print(f'Acc:{accuracy_score(y_true, y_pred)}')
    ################ precision ##############
    print(f'Precision:{precision_score(y_true, y_pred, average="macro")}')
    ################ Recall ##############
    print(f'Recall:{recall_score(y_true, y_pred, average="macro")}')
    ################ F1 ##############
    print(f'F1-Score:{f1_score(y_true, y_pred, average="macro")}')
    return fitted_model, pred