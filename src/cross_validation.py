from preprocessing import load_prepocessed_data
from model import create_model

from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# stops loggs from tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def cross_validate():
    X, y = load_prepocessed_data()

    params = {
        'epochs': [10, 20],
        'batch_size': [32, 64, 128],
        'model__conv_layer_num': [2, 3], 
        'model__filter_num': [3], 
        'model__kernel_size': [(8,8)], 
        'model__pool_size': [(2,2)], 
        'model__dropout_coeff_conv': [0.5], 
        'model__dense_layer_num': [1,2], 
        'model__units_num': [64, 128], 
        'model__dropout_coeff_dense': [0.2]
    }
    
    model = KerasClassifier(model=create_model, verbose=0)
    gs = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=3, verbose=3)
    
    grid_result = gs.fit(X, y)
    
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == '__main__':
    cross_validate()