
import random
import traceback
from pdb import set_trace

import numpy as np

from itcs4156.util.timer import Timer
from itcs4156.util.eval import RunModel
from itcs4156.util.metrics import accuracy
from itcs4156.assignments.classification.LogisticRegression import LogisticRegression
from itcs4156.assignments.classification.Perceptron import Perceptron
from itcs4156.assignments.classification.NaiveBayes import NaiveBayes
from itcs4156.assignments.classification.train import HyperParametersAndTransforms as hpt
from itcs4156.datasets.DataPreparation import MNISTDataPreparation

def rubric_perceptron(acc, max_score=25):
    score_percent = 0
    if acc >= 0.95:
        score_percent = 100
    elif acc >= 0.90:
        score_percent = 90
    elif acc >= 0.80:
        score_percent = 80
    elif acc >= 0.70:
        score_percent = 70
    elif acc >= 0.60:
        score_percent = 60
    elif acc >= 0.50:
        score_percent = 50
    else:
        score_percent = 40
    score = max_score * score_percent / 100.0 
    return score

def rubric_naive_bayes(acc, max_score=25):
    score_percent = 0
    if acc >= 0.75:
        score_percent = 100
    elif acc >= 0.65:
        score_percent = 90
    elif acc >= 0.55:
        score_percent = 80
    elif acc >= 0.40:
        score_percent = 70
    elif acc >= 0.30:
        score_percent = 60
    elif acc >= 0.20:
        score_percent = 50
    elif acc >= 0.10:
        score_percent = 45
    else:
        score_percent = 40
    score = max_score * score_percent / 100.0 
    return score
   
def rubric_logistic_regression(acc, max_score=30):
    score_percent = 0
    if acc >= 0.85:
        score_percent = 100
    elif acc >= 0.80:
        score_percent = 90
    elif acc >= 0.75:
        score_percent = 80
    elif acc >= 0.70:
        score_percent = 70
    elif acc >= 0.60:
        score_percent = 60
    elif acc >= 0.50:
        score_percent = 55
    elif acc >= 0.40:
        score_percent = 50
    elif acc >= 0.30:
        score_percent = 45
    else:
        score_percent = 40
    score = max_score * score_percent / 100.0 
    return score


def run_eval(eval_stage='validation'):
    main_timer = Timer()
    main_timer.start()

    total_points = 0
    
    task_info = [
       dict(
            model=Perceptron,
            name='Perceptron',
            data=MNISTDataPreparation,
            data_prep=dict(binarize=True, return_array=True),
            metrics=dict(acc=accuracy),
            eval_metric='acc',
            rubric=rubric_perceptron,
            trn_score=0,
            eval_score=0,
            successful=False,
        ),
        dict(
            model=NaiveBayes,
            name='NaiveBayes',
            data=MNISTDataPreparation,
            data_prep=dict(return_array=True),
            metrics=dict(acc=accuracy),
            eval_metric='acc',
            rubric=rubric_naive_bayes,
            trn_score=0,
            eval_score=0,
            successful=False,
        ),
        dict(
            model=LogisticRegression,
            name='LogisticRegression',
            data=MNISTDataPreparation,
            data_prep=dict(return_array=True),
            metrics=dict(acc=accuracy),
            rubric=rubric_logistic_regression,
            eval_metric='acc',
            trn_score=0,
            eval_score=0,
            successful=False,
        ),
    ]
    
    total_points = 0
    for info in task_info:
        task_timer =  Timer()
        task_timer.start()
        try:
            params = hpt.get_params(info['name'])
            model_kwargs = params.get('model_kwargs', {})
            data_prep_kwargs = params.get('data_prep_kwargs', {})
            
            run_model = RunModel(info['model'], model_kwargs)
            data = info['data'](**data_prep_kwargs)
            X_trn, y_trn, X_vld, y_vld = data.data_prep(**info['data_prep'])
            
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
        points = info['rubric'](info['eval_score'])
        print(f"Points Earned: {points}")
        total_points += points
        
    print("="*50)
    print('')
    main_timer.stop()
    
    avg_trn_acc, avg_eval_acc, successful_tests = summary(task_info)
    task_eval_acc = get_eval_scores(task_info)
    total_points = int(round(total_points))
    
    print(f"Tests passed: {successful_tests}/{ len(task_info)}, Total Points: {total_points}/80\n")
    print(f"Average Train Accuracy: {avg_trn_acc}")
    print(f"Average {eval_stage.capitalize()} Accuracy: {avg_eval_acc}")
    
    return (total_points, avg_eval_acc, main_timer.last_elapsed_time, avg_trn_acc, *task_eval_acc)

def summary(task_info):
    sum_trn_acc = 0
    sum_eval_acc = 0
    successful_tests = 0

    for info in task_info:
        if info['successful']:
            successful_tests += 1
            sum_trn_acc += info['trn_score']
            sum_eval_acc += info['eval_score']
    
    if successful_tests == 0:
        return 0, 0, successful_tests
    
    avg_trn_acc = sum_trn_acc / len(task_info)
    avg_eval_acc = sum_eval_acc / len(task_info)
    return round(avg_trn_acc, 4), round(avg_eval_acc, 4), successful_tests

def get_eval_scores(task_info):
    return [i['eval_score'] for i in task_info]

if __name__ == "__main__":
    run_eval()

