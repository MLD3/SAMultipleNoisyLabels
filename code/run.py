import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import random
import pickle
from datetime import datetime
import os
import sys
import argparse

import get_data
import util
from hyperparams import all_hyperparams, hp_ranges
from data_settings import all_settings 
from exp_settings import all_exp_settings

from directories import results_dir

seed = 123456789

###################################################################################################
'''
run an experiment with a specific approach on one dataset with one approach
'''
def run_exp(dataset_name, data_package, approach, tune, split_seed=0, date='0', coef=0, out_var=0, ncoef=0, neg_thresh=0):
    #random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
   
    if not os.path.isdir(results_dir + dataset_name + '/'):
        os.mkdir(results_dir + dataset_name + '/')
    
    data_params = all_settings[dataset_name]
    data_params['num_feats'] = data_package[0]['feats'].shape[1]
    data_params['num_steps'] = data_package[0]['horizon'] 
    data_params['coef'] = coef
    data_params['out_var'] = out_var
    data_params['ncoef'] = ncoef
    data_params['neg_thresh'] = neg_thresh
    
    approach_name = approach
    if 'baseline_plain' in approach:
        approach_name = 'baseline_plain'
    elif 'proposed3' in approach:
        approach_name = 'proposed3'
    hyperparams = all_hyperparams[dataset_name][approach_name]
    hyperparam_ranges = hp_ranges[dataset_name][approach_name]
    
    if tune:
        mod, hyperparams, res = util.tune_hyperparams(data_package, approach, data_params, \
                                hyperparam_ranges, results_dir, date, dataset_name)
        print(dataset_name, approach, hyperparams, hyperparam_ranges)
        print('test results \n', res['c index'])
        return res
        
    else:
        mod, _, _ = util.get_model(dataset_name, data_package, approach, data_params, hyperparams)
        test_data = data_package['test']
        test_cov, test_times, test_cen_in = test_data['feats'], test_data['gt_obs_times'], test_data['gt_cen_in']
        test_cov = util.to_gpu([test_cov])
        test_out = mod(test_cov).detach().cpu().numpy()          
        test_eval = evaluate(test_out, test_times, test_cen_in)


'''
run an experiment with a specific setting on one dataset with multiple approaches
'''
def run_bulk_exp(dataset_name, approaches, tune, split_seed=0, date='0'):
    #random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    time_now = datetime.now()

    data_params = all_settings[dataset_name]
    data_package, coef, out_var, ncoef, neg_thresh = get_data.get_dataset(dataset_name, data_params, split_seed=split_seed)
    
    num_labels = data_params['num_labelers']
    
    all_results = {}
    for approach in approaches:
        res = run_exp(dataset_name, data_package, approach, tune, split_seed, date, coef, out_var, ncoef, neg_thresh)
        all_results[approach] = res
        
    end_time = datetime.now()
    print('ran on 1 setting, time elapsed: ', end_time - time_now)
    return all_results
    
    
'''
run an experiment on multiple settings
'''
def run_mult_settings(exp_settings, data_params, dataset_name, approaches, tune, split_seeds):
    all_results = []
    
    for setting in exp_settings:
        for key in list(setting.keys()):
            data_params[key] = setting[key]
    
        setting_res = {}
        for i in range(len(split_seeds)):
            split_seed = seed + split_seeds[i]
            seeded_date = date + '_' + str(split_seeds[i])
            res = run_bulk_exp(dataset_name, approaches, tune, split_seed, seeded_date)
            setting_res[split_seeds[i]] = res
            
        all_results.append(setting_res)
            
    return all_results
    

'''
display results
'''
def display_results(results, dataset_name, exp_name, exp_settings):
    print('**********************************************************************')
    print(dataset_name, exp_name)
    for i in range(len(exp_settings)):
        print('experimental conditions:', exp_settings[i])  
        overall_res = {}
        for seed in list(results[i].keys()):
            for approach in list(results[i][seed].keys()):
                appr_res = results[i][seed][approach]
                for metric in list(appr_res.keys()):
                    if approach not in list(overall_res.keys()):
                        overall_res[approach] = {}
                        overall_res[approach][metric] = [appr_res[metric]]
                    elif metric not in list(overall_res[approach].keys()):
                        overall_res[approach][metric] = [appr_res[metric]]
                    else:
                        overall_res[approach][metric].append(appr_res[metric])
        plt.figure(figsize=(30, 4.5))
        for j in range(len(list(overall_res.keys()))):
            approach = list(overall_res.keys())[j]
            print(approach)
            print('c index', overall_res[approach]['c index'][0])
            if dataset_name == 'synth':
                print('average bs preds', np.percentile(overall_res[approach]['bs'][0]['preds'], [0, 25, 50, 75, 100]))
            else:
                print('ddc', overall_res[approach]['ddc'][0])
            print('median - tte', np.percentile(overall_res[approach]['predicted median tte minus gt tte'][0], [0, 25, 50, 75, 100]))
            print('censored median accuracy', overall_res[approach]['censored median accuracy'][0])
      
        print('\n')
        print('##########################################')
        
    return 1
    
###################################################################################################
'''
main block

approaches: 
    baseline_plain_{avg, avgvote} -> aggregation (Naive average and voting average)
    baseline_separate -> learn a model for each labeler separately (Independent)
    proposed3_obs -> learn observed label distr only with just step 1a (Repeated Samples), removed from paper due to redundancy with margerr
    proposed3 -> learn observed label distr and gt distr with proposed loss (Proposed -Error)
    proposed3_noneg -> learn observed label distr, gt distr, and error distr -supervision on negatives (Proposed -Negative)
    proposed3_err -> learn observed label distr, gt distr, and error distr with supervision on negatives (Proposed)
    proposed3_margerr -> learn observed label distr, error distr, and derive gt distr by summing and marginalizing (Proposed -Recalibration)

datasets:
    synth (Synthetic data)
    mimic (MIMIC-III for sepsis prediction)
'''
if __name__ == '__main__':
    #random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    time_now = datetime.now()
    date = '241225'#set this however
    tune = True

    parser = argparse.ArgumentParser(description='Datasets and experimental conditions for noisy labels experiments')
    parser.add_argument('--dataset', default='synth')
    parser.add_argument('--experiment', default='baseline')
    #same_error is always false since noise is instance-dependent (i.e., the error is not the same across instances)
    parser.add_argument('--same_error', default='False') 
    args = parser.parse_args()
    
    split_seeds = [0,1,2,3,4,5,6,7,8,9]

    dataset_name = args.dataset
    data_params = all_settings[dataset_name]
    data_params['same_error'] = args.same_error == 'True'
    
    approaches1 = ['baseline_plain_avgnaive', 'baseline_plain_avgvote', \
                   'baseline_separate', 'proposed3_obs', 'proposed3'] 
    approaches2 = ['proposed3_margerr', 'proposed3', 'proposed3_err']
    approaches3 = ['proposed3_noneg', 'proposed3_err']
    approaches_mimic = ['baseline_plain_avgnaive', 'baseline_plain_avgvote', \
                   'baseline_separate', 'proposed3_obs', 'proposed3_err']
    approaches = approaches_mimic
    
    exp_name = args.experiment
    if exp_name == 'pcen': #see how performance changes with proportion partially censored
        approaches = approaches1
        exp_settings = all_exp_settings[dataset_name][0:9][::-1] 
    elif exp_name == 'pcen100000': #see how performance changes for proposed approach with large training size
        approaches = approaches1[-1:]
        exp_settings = all_exp_settings[dataset_name][0:9]
        for setting in exp_settings:
            setting['train_size'] =  100000 
            setting['num_data'] = 102250
    elif exp_name == 'offset': #see how performance changes with differnt noise means 
        approaches = approaches2
        exp_settings = all_exp_settings[dataset_name][10:21] 
    elif exp_name == 'neg': #see how performance changes with different noise rates
        approaches = approaches3
        exp_settings = all_exp_settings[dataset_name][21:31][::-1]
    elif exp_name == 'size': #see how performance changes with different anchor sizes
        approaches = ['proposed3_err']
        exp_settings = all_exp_settings[dataset_name][1:11]
    else:
        split_seeds = [0,1,2,3,4,5,6,7,8,9]
        exp_settings = all_exp_settings[dataset_name][:1] 
    
    print('approaches: ', approaches)
    
    results = run_mult_settings(exp_settings, data_params, dataset_name, approaches, tune, split_seeds)
         
    display_results(results, dataset_name, exp_name, exp_settings)
     
    end_time = datetime.now()
    print('overall experiment, time elapsed: ', end_time - time_now)
    
