import os
import random
import traceback
from pdb import set_trace

import numpy as np

from itcs4156.util.eval import RunModel
from itcs4156.util.timer import Timer
from itcs4156.util.data import split_data, feature_label_split, Standardization
from itcs4156.util.metrics import mse
from itcs4156.datasets.HousingDataset import HousingDataset

from itcs4156.assignments.regression.OrdinaryLeastSquares import OrdinaryLeastSquares
from itcs4156.assignments.regression.LeastMeanSquares import LeastMeanSquares
from itcs4156.assignments.regression.PolynomialRegression import PolynomialRegression
from itcs4156.assignments.regression.PolynomialRegressionRegularized import PolynomialRegressionRegularized
from itcs4156.assignments.regression.train import HyperParameters

def standardize_data(X_trn, X_vld):
    standardize = Standardization()
    X_trn_clean = standardize.fit_transform(X_trn)
    X_eval_clean = standardize.transform(X_vld)
    
    return X_trn_clean, X_eval_clean

def get_cleaned_data(df_trn, df_vld, feature_names, label_name, return_df=False):
    X_trn, y_trn, X_vld, y_vld = split_data(df_trn, df_vld, feature_names, label_name)
    X_trn, X_vld = standardize_data(X_trn, X_vld)

    return X_trn, y_trn, X_vld, y_vld

def run_eval(eval_stage='validation'):
    main_timer = Timer()
    main_timer.start()

    # set_seeds(seed=25)

    dataset = HousingDataset()
    df_trn, df_vld = dataset.load()

    total_points = 0
    
    task_info = [
       dict(
            model=OrdinaryLeastSquares,
            name='OrdinaryLeastSquares',
            threshold=50,
            metrics=dict(MSE=mse),
            eval_metric='MSE',
            trn_score=9999,
            eval_score=9999,
            successful=False,
        ),
        dict(
            model=LeastMeanSquares,
            name='LeastMeanSquares',
            threshold=50,
            metrics=dict(MSE=mse),
            eval_metric='MSE',
            trn_score=9999,
            eval_score=9999,
            successful=False,
        ),
        dict(
            model=PolynomialRegression,
            name='PolynomialRegression',
            threshold=30,
            metrics=dict(MSE=mse),
            eval_metric='MSE',
            trn_score=9999,
            eval_score=9999,
            successful=False,
        ),
        dict(
            model=PolynomialRegressionRegularized,
            name='PolynomialRegressionRegularized',
            threshold=20,
            metrics=dict(MSE=mse),
            eval_metric='MSE',
            trn_score=9999,
            eval_score=9999,
            successful=False,
        )
    ]
    
    for info in task_info:
        task_timer =  Timer()
        task_timer.start()
        try: 
            params = HyperParameters.get_params(info['name'])
            model_kwargs = params.get('model_kwargs', {})
            data_prep_kwargs = params.get('data_prep_kwargs', {})

            if info['name'] == 'OrdinaryLeastSquares':
                feature_names = "RM"
            elif info['name'] == 'PolynomialRegression':
                feature_names = "LSTAT"
            else:
                use_features = data_prep_kwargs.get('use_features')
                if use_features is None:
                    err = f"use_features argument for {info['name']} can not be none: received {use_features}"
                    raise ValueError(err)
                elif  len(use_features) < 2 :
                    err = f"use_features argument for {info['name']} must have at least 2 features: received {use_features}"
                    raise ValueError(err)
                
                feature_names = data_prep_kwargs['use_features']

            run_model = RunModel(info['model'], model_kwargs)
            
            # if "MEDV" in feature_names:
            #     print("\nThe target feature 'MEDV' can not be used as an input feature!")
            #     print("Removing MEDV from your feature list and proceeding...\n")
            #     feature_names = [ f for f in feature_names if f != "MEDV"]
            
            X_trn, y_trn, X_vld, y_vld = get_cleaned_data(df_trn, df_vld, feature_names, "MEDV")

            trn_scores = run_model.fit(X_trn, y_trn, info['metrics'], pass_y=True)
            eval_scores = run_model.evaluate(X_vld, y_vld, info['metrics'], prefix=eval_stage.capitalize())

            info['trn_score'] = trn_scores[info['eval_metric']]
            info['eval_score'] = eval_scores[info['eval_metric']]
            info['successful'] = True
                
        except Exception as e:
            track = traceback.format_exc()
            print("The following exception occurred while executing this test case:\n", track)
        task_timer.stop()
        
        print("")
        points = rubric_regression(info['eval_score'], info['threshold'])
        print(f"Points Earned: {points}")
        total_points += points

    print("="*50)
    print('')
    main_timer.stop()

    avg_trn_mse, avg_eval_mse, successful_tests = summary(task_info)
    task_eval_mse = get_eval_scores(task_info)
    total_points = int(round(total_points))

    print("Tests passed: {}/4, Total Points: {}/80\n".format(successful_tests, total_points))
    print(f"MSE averages for {successful_tests} successful tests")
    print(f"\tAverage Train MSE: {avg_trn_mse}")
    print(f"\tAverage {eval_stage.capitalize()} MSE: {avg_eval_mse}")
    
    return (total_points, avg_trn_mse, avg_eval_mse, *task_eval_mse)

def rubric_regression(mse, thresh, max_score=20):
    if mse <= thresh:
        score_percent = 100
    elif mse is not None:
        score_percent = (thresh / mse) * 100
        if score_percent < 40:
            score_percent = 40
    else:
        score_percent = 20
    score = max_score * score_percent / 100.0

    return score

def get_eval_scores(task_info):
    return [i['eval_score'] for i in task_info]

def summary(task_info):
    sum_trn_mse = 0
    sum_eval_mse = 0
    successful_tests = 0

    for info in task_info:
        if info['successful']:
            successful_tests += 1
            sum_trn_mse += info['trn_score']
            sum_eval_mse += info['eval_score']
    
    if successful_tests == 0:
        return 9999, 9999, successful_tests
    
    avg_trn_mse = sum_trn_mse / successful_tests
    avg_eval_mse = sum_eval_mse / successful_tests
    return avg_trn_mse, avg_eval_mse, successful_tests


if __name__ == "__main__":
    run_eval()
  




    


