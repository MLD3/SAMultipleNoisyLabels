'''
code to load, generate and split data lives here
'''

import numpy as np
import pandas as pd
import scipy.sparse
import torch
import sparse

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import copy
import random
import pickle

from directories import mimic_root

################################################################################################################
'''
reads file and turns to numpy array
'''
def get_file(file_name, dim):
    f = open(file_name, 'r')
    c = f.read()
    c = c[1:]
    c = c.replace('\n', ',')
    c = c.split(',')
    c = np.array(c)
    c = c[:-1]
    c = c.reshape((-1,dim))
    f.close()
    return c


'''
normalize data to 0-1 range
'''
def normalize(data):
    mins = np.amin(data, axis=0)
    maxs = np.amax(data, axis=0)
    
    dims = data.shape
    mins = np.tile(mins, (dims[0], 1))
    maxs = np.tile(maxs, (dims[0], 1))
    
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    
    data[:, :] = (data[:,:] - mins[:, :]) / (ranges)
    return data


################################################################################################################
'''
generates a noisy observed time to event
'''        
def get_noisy_time(feats, outcomes, bias, same_error, offset, error_sum, last):
    n_outcome = np.zeros((feats.shape[0],))
    print('n_outcome distr', np.percentile(n_outcome, [0, 5, 25, 50, 75, 95, 100]))
    
    noise_sd = 0
    proxy_out = np.zeros((feats.shape[0],))
    for i in range(feats.shape[0]):
        noisy_time = np.random.normal(outcomes[i] + n_outcome[i] + bias + offset, noise_sd)
        if not same_error and int(outcomes[i]//20) % 2 == 0:
            noisy_time = np.random.normal(outcomes[i] - n_outcome[i] - bias + offset, noise_sd)
        proxy_out[i] = max(0, noisy_time)
    return proxy_out.astype(int), n_outcome
    
'''
censor at random and at the prediction horizon

full censoring - number of individuals whose followup time is before ground truth and any observed time
general censoring - number of individuals whose followup is after ground truth but excludes some observeds
''' 
def censor(feats, times, noisy_times, cen_prop, horizon):
    num_ex = times.shape[0]
    cen_prop_f, cen_prop_g = cen_prop[0], cen_prop[1]
    num_cen = int(num_ex * (cen_prop_g + cen_prop_f))
    num_cen_full = int(num_ex * cen_prop_f)
    
    obs_times = copy.deepcopy(times)
    last_times = np.zeros(times.shape)
    cens_in = np.zeros(times.shape)
    
    perm = np.random.permutation(num_ex)
    
    full_cens = perm[:num_cen_full]
    for ex in full_cens:
        obs_ex = [noisy_times[i][ex] for i in range(len(noisy_times))]
        min_time = min(min(obs_ex), times[ex])
        new_time = np.random.randint(0, max(min_time-1, 0)+1)
        
        obs_times[ex] = new_time
        last_times[ex] = new_time
        cens_in[ex] = 1
    
    part_cens = perm[num_cen_full:num_cen]
    num_cen_lab = []
    for ex in part_cens:
        obs_ex = np.sort([noisy_times[i][ex] for i in range(len(noisy_times))])
        obs_ex2 = obs_ex[obs_ex <= horizon-1]
        if obs_ex[-1] == obs_ex[0]:
            obs_ex[-1] += 1
        new_time = min(np.random.randint(obs_ex[0], obs_ex[-1]), horizon-1)
        cen_labs = np.where(np.array(obs_ex)>new_time)[0]
        num_cen_lab.append(cen_labs.shape[0])
        last_times[ex] = new_time
        
        if new_time < obs_times[ex]:
            obs_times[ex] = new_time
            cens_in[ex] = 1
    
    uncen = perm[num_cen:]
    for ex in uncen:
        last_times[ex] = horizon-1 
    
    h_cen = np.where(obs_times >= horizon)[0]   
    obs_times[h_cen] = horizon - 1
    cens_in[h_cen] = 1
    last_times[h_cen] = horizon - 1
    
    return obs_times, last_times.astype(int), cens_in
    
    
'''
assign censorship status - get the censorship indicator variable
'''
def assign_censorship(event_times, obs_times, last_time, cen_i):
    new_times = copy.deepcopy(event_times)
    cens_in = np.zeros(event_times.shape)
    
    censored = np.where(event_times > last_time)[0]
    if censored.shape[0] > 0:
        cens_in[censored] = 1
        new_times[censored] = last_time[censored]
    
    return new_times, cens_in
    
    
''''
add noise to the dataset 
'''
def add_noise(feats, times, args, horizon):
    num_labelers = args['num_labelers']
    cen_prop = args['cen_prop']
    same_error = args['same_error']
    
    bias = 0
    if num_labelers == 3:
        bias = np.array([-20, -20, 40])
    elif num_labelers == 2:
        bias = np.array([-20, 20]) 
    elif num_labelers == 1:
        bias = np.array([20]) 
    
    if args['offset'] == 'random':
        bias = np.random.normal(0, 5, size=(num_labelers,))
        bias[bias < 0] = np.floor(bias[bias < 0])
        bias[bias > 0] = np.ceil(bias[bias > 0])
    
    print('noise biases', bias)
    noisy_times = [] #time to event each labeler would assign
    error_sum = np.zeros((feats.shape[0],))
    for i in range(num_labelers):
        noisy_time, errors = get_noisy_time(feats, times, bias[i], same_error, args['offset'], error_sum, i==num_labelers-1)
        error_sum += errors
        noisy_times.append(noisy_time.astype(int))
    
    #observed in dataset
    obs_time, last_time, gt_cen_in = censor(feats, times, noisy_times, cen_prop, horizon) 
    noisy_obs_time, cen_in = [], []
    for i in range(num_labelers):
        new_obs_time, cen_in_i = assign_censorship(noisy_times[i], obs_time, last_time, gt_cen_in)
        cen_in.append(cen_in_i.astype(int))
        noisy_obs_time.append(new_obs_time.astype(int))
    
    return noisy_obs_time, cen_in, [obs_time], [gt_cen_in], last_time
    
'''
make it so that some people never get the event (true negative examples)
and also maybe false positives
'''
def add_negs(feats, obs_times, cen_in, gt_obs_time, gt_cen_in, gt_time, prop_neg, noise_rates, horizon):
    coef = np.random.normal(size=(feats.shape[1]))
    neg_i = np.sum(coef * feats, axis=1)
    neg_thresh = np.percentile(neg_i, prop_neg*100) 
    
    #anything below the xth percentile is negative
    neg_ex = np.where(neg_i < neg_thresh)[0]   
    
    fp_i, fn_i = [], []
    for i in range(len(obs_times)): 
        fp_coef = np.random.normal(size=(feats.shape[1]))
        fp_i.append(np.sum(fp_coef*feats, axis=1))
        fn_coef = np.random.normal(size=(feats.shape[1]))
        fn_i.append(np.sum(fn_coef*feats, axis=1))
   
    for i in range(len(neg_i)):
        obs_i = np.array([obs_times[j][i] for j in range(len(obs_times))])
        cen_i = np.array([cen_in[j][i] for j in range(len(cen_in))])
        for j in range(len(obs_i)):
            #fp, fn rates
            fp_rate, fn_rate = noise_rates[j][0], noise_rates[j][1]
            fp_thresh = np.percentile(fp_i[j], fp_rate*100)
            fn_thresh = np.percentile(fn_i[j], fn_rate*100)
    
            noisy_fp = 1 if fp_i[j][i] < fp_thresh else 0
            noisy_fn = 1 if fn_i[j][i] < fn_thresh else 0
            #recorded as negative (true and false negatives)
            if (neg_i[i] < neg_thresh and noisy_fp == 0) or (neg_i[i] >= neg_thresh and noisy_fn == 1):
                if cen_i[j] == 1:
                    continue
                if 1 in cen_i:
                    last_seen = obs_i[np.where(cen_i == 1)[0][0]]
                    obs_times[j][i] = last_seen    
                else:
                    obs_times[j][i] = horizon-1
                cen_in[j][i] = 1
             
    gt_obs_time[0][neg_ex] = horizon-1
    gt_cen_in[0][neg_ex] = 1
    gt_time[neg_ex] = horizon-1
    
    return obs_times, cen_in, gt_obs_time, gt_cen_in, gt_time, coef, neg_thresh
       
################################################################################################################ 
'''
synthetic data
'''
def get_synth(args):
    num_data, num_feats = args['num_data'], args['num_feats']
    horizon = args['horizon']
    prop_neg = args['prop_neg'] if 'prop_neg' in list(args.keys()) else 0
    noise_rates = args['noise'] if 'noise' in list(args.keys()) else 0
    
    feats = np.random.normal(0, 1, size=(num_data, num_feats))
    
    coef = np.random.uniform(0, 1, size=(feats.shape[1]))
    outcome = coef * feats
    outcome = np.sum(outcome, axis=1)
    
    outcome = (outcome - np.min(outcome)) / (np.max(outcome) - np.min(outcome))
    outcome = np.floor(outcome*(horizon-1)).astype(int)
    
    out_var = 2
    for i in range(len(outcome)):
        outcome[i] = min(int(np.random.normal(outcome[i], out_var)), horizon - 1) 
        outcome[i] = max(outcome[i], 0)
    
    obs_times, cen_in, gt_obs_time, gt_cen_in, last_time = add_noise(feats, copy.deepcopy(outcome), args, horizon)
    
    ncoef, neg_thresh = 0, 0
    if prop_neg > 0:
        obs_times, cen_in, gt_obs_time, gt_cen_in, outcome, ncoef, neg_thresh = add_negs(feats, obs_times, cen_in, gt_obs_time, \
            gt_cen_in, outcome, prop_neg, noise_rates, horizon)
    
    print('observed censor rates')
    for i in range(len(cen_in)):
        print(np.unique(cen_in[i], return_counts=True))
    print('ground truth tte distr', np.unique(outcome, return_counts=True))
    
    num_cen_labs = []
    for i in range(num_data):
        num_censored = 0
        for j in range(len(cen_in)):
            num_censored += cen_in[j][i]
        num_cen_labs.append(num_censored)
    num_cen_labs = np.array(num_cen_labs)
    print('number of censored labels per instance [0, 25, 50, 75, 100] percentiles')
    print(np.percentile(num_cen_labs, [0, 25, 50, 75, 100]))
    have_event = np.where(gt_cen_in[0] == 0)[0]
    num_cen_with_e = num_cen_labs[have_event]
    print('prop fully cen', np.where(num_cen_with_e[num_cen_with_e == args['num_labelers']])[0].shape[0] / have_event.shape[0])
    print('prop part cen', np.where(num_cen_with_e[num_cen_with_e>0] < args['num_labelers'])[0].shape[0] / have_event.shape[0])
    print('censored', np.where(num_cen_labs > 0)[0].shape[0] / num_data)
    
    print('horizon:', horizon, 'prop gt cen', np.sum(gt_cen_in[0])/gt_cen_in[0].shape[0])
    dataset = {'feats': feats, 'gt_times': outcome, \
               'obs_times': obs_times, 'cen_in': cen_in, 'horizon': horizon, \
               'gt_obs_times': gt_obs_time, 'gt_cen_in': gt_cen_in}
    
    return dataset, coef, out_var, ncoef, neg_thresh, last_time
  
  
'''
get labeler stats
'''
def get_labeler_stats(obs, obs_cen, gt, gt_cen, horizon):
    print('observed sepsis rate', 1 - np.sum(obs_cen)/obs_cen.shape[0], obs_cen.shape[0]-np.sum(obs_cen))
    gt_pos = np.where(gt_cen == 0)[0]
    obs_pos = np.where(obs_cen == 0)
    gt_neg = np.intersect1d(np.where(gt_cen == 1)[0], np.where(gt == horizon-1)[0])
    obs_neg = np.intersect1d(np.where(obs_cen == 1)[0], np.where(obs == horizon-1)[0])
    
    tp = np.intersect1d(gt_pos, obs_pos).shape[0]
    fp = np.intersect1d(gt_neg, obs_pos).shape[0]
    tn = np.intersect1d(gt_neg, obs_neg).shape[0]
    fn = np.intersect1d(gt_pos, obs_neg).shape[0]
    
    print('true positive rate', tp / (tp + fn), tp)
    print('false positive rate', fp / (tn + fp), fp)
    print('true negative rate', tn / (tn + fp), tn)
    print('false negative rate', fn / (tp + fn), fn)
    print('censored', 1 - ((tp + fp + tn + fn)/gt.shape[0]), gt.shape[0]-(tp+fp+tn+fn))
    
    noise_err = obs[np.intersect1d(gt_pos, obs_pos)] - gt[np.intersect1d(gt_pos, obs_pos)]
    print('noise error 0, 25, 50, 75, 100 percentiles', np.percentile(noise_err, [0, 25, 50, 75, 100]))
     
  
'''
real data - sepsis
label columns: 'HADM_ID', 'sepsis1 time', 'sepsis3 time', 'sepsis comp', 'last obs'

the current composite definition is like an 'or' -> try an 'and'
the 'or' picks up whatever labeler is assigned and since there isn't a lot of overlap, don't run into averaging/censoring issues
having an 'and' based definition may help introduce these issues
'''
def get_mimic(args):
    pop_file = mimic_root + 'fiddle_output/IDs.csv'
    pop = pd.read_csv(pop_file)
    lab_file = mimic_root + 'labels_composite_cmscdc.csv'
    print('using ground truth from', lab_file)
    labels = pd.read_csv(lab_file)
    extra_labs = pd.read_csv(mimic_root + 'labels_composite.csv')
    extra_labs['sep1_extra'] = extra_labs['sepsis1 time']
    extra_labs['sep3_extra'] = extra_labs['sepsis3 time']
    extra_labs['sep_comp_extra'] = extra_labs['sepsis comp']
    labels = labels.merge(extra_labs[['HADM_ID', 'sep1_extra', 'sep3_extra', 'sep_comp_extra']], on='HADM_ID')
    
    num_both = 0
    for i in range(labels.shape[0]):
        if labels['sepsis1 time'].iloc[i] != 'None' and labels['sepsis3 time'].iloc[i] != 'None': 
            num_both += 1
    print(num_both)
    labels = pop.merge(labels, left_on='ID', right_on='HADM_ID', how='left')
    print('labels', labels.shape)
    num_both = 0
    for i in range(labels.shape[0]):
        if labels['sepsis1 time'].iloc[i] != 'None' and labels['sepsis3 time'].iloc[i] != 'None': 
            num_both += 1
    print(num_both)
    
    time_var = mimic_root + 'fiddle_output/X.npz'
    feats_var = np.load(time_var)
    feats_var = sparse.COO(feats_var['coords'], feats_var['data'], tuple(feats_var['shape']))
    
    time_invar = mimic_root + 'fiddle_output/S.npz'
    feats_invar = np.load(time_invar)
    feats_invar = sparse.COO(feats_invar['coords'], feats_invar['data'], tuple(feats_invar['shape']))
    
    feats = feats_invar
    print('time varying, time invarying', feats_var.shape, feats_invar.shape)
    for i in range(feats_var.shape[1]):
        feats = np.concatenate((feats, feats_var[:, i, :]), axis=1)   
    num_feat = feats.shape[1]
    feats = feats.todense()
    print('features', feats.shape)
    
    horizon = args['horizon']
    last_time = labels['last obs'].to_numpy()
    last_time[last_time > horizon] = horizon - 1
    
    gt_lab = 'sepsis comp'
    gt_obs_times, gt_cen_in = np.zeros((pop.shape[0],)), np.zeros((pop.shape[0],))
    gt_obs_times[labels[gt_lab] != 'None'] = labels[gt_lab][labels[gt_lab] != 'None'].to_numpy().astype(int)
    gt_obs_times[labels[gt_lab] == 'None'] = horizon
    gt_cen_in[labels[gt_lab] == 'None'] = 1
    gt_cen_in[gt_obs_times >= horizon] = 1
    gt_obs_times[gt_obs_times >= horizon] = horizon - 1
    
    obs_times = []
    cen_in = []
    
    #obs_names = ['sepsis1 time', 'sepsis3 time', 'sep1_extra', 'sep3_extra', 'sep_comp_extra'] #use for getting stats only
    obs_names = ['sep3_extra', 'sep1_extra', 'sep_comp_extra'] #final use
    
    for name in obs_names:
        sep_times, sep_cen_in = np.zeros((pop.shape[0],)), np.zeros((pop.shape[0],))
        sep_times[labels[name] != 'None'] = labels[name][labels[name] != 'None'].to_numpy().astype(int)
        sep_times[labels[name] == 'None'] = last_time[labels[name] == 'None']#horizon - 1
        sep_cen_in[labels[name] == 'None'] = 1
        sep_cen_in[sep_times >= horizon] = 1
        sep_times[sep_times >= horizon] = horizon - 1
        obs_times.append(sep_times)
        cen_in.append(sep_cen_in)
    
    print('horizon:', horizon, 'prop gt cen', np.sum(gt_cen_in)/gt_cen_in.shape[0])
    for i in range(len(obs_names)):
        print(obs_names[i] + ' stats')
        get_labeler_stats(obs_times[i], cen_in[i], gt_obs_times, gt_cen_in, horizon)
    print('sepsis gt rate', 1  -np.sum(gt_cen_in)/gt_cen_in.shape[0], np.where(gt_cen_in == 0)[0].shape[0])
    print('sepsis gt censored', np.intersect1d(np.where(gt_cen_in == 1)[0], np.where(gt_obs_times < horizon)[0]).shape[0]/pop.shape[0])
    print('sepsis gt tte distr', np.percentile(gt_obs_times[gt_cen_in == 0], [0, 25, 50, 75, 100]))
    
    dataset = {'feats': feats, 'gt_times': gt_obs_times, \
               'obs_times': obs_times, 'cen_in': cen_in, 'horizon': horizon, \
               'gt_obs_times': [gt_obs_times], 'gt_cen_in': [gt_cen_in]}
    
    return dataset, 0, 0, 0, 0, last_time


################################################################################################################
'''
return a subset of the data at the specified indexes
'''
def get_subset(raw_data, indexes):
    keys = list(raw_data.keys())
    subset_data = {}
    
    for key in keys:
        original = raw_data[key]
        if isinstance(original, list):
            subset = []
            for item in original:
                item_tensor = torch.Tensor(item[indexes])
                subset.append(item_tensor)
            subset_data[key] = subset
        elif isinstance(original, int):
            subset_data[key] = original
        elif len(original.shape) == 1:
            subset_data[key] = torch.Tensor(original[indexes])
        elif len(original.shape) == 2:
            subset_data[key] = torch.Tensor(original[indexes, :])

    return subset_data
    
    
'''
split data into training, validation, and test sets
correctly labeled data can go to training set (correct_where='train') or validation (correct_where='val')
    might want to look at putting a proportion into both...
'''
def split_data(raw_data, split_seed, args, last_time):  
    np.random.seed(123456789 + split_seed)
    random.seed(123456789 + split_seed)
    
    feats = raw_data['feats']
    train_size = args['train_size']
    anc_size = args['anc_size']
    num_val = 250 #number in validation set

    #split into training/not training
    prop_not_train = (num_val + 2000) / raw_data['feats'].shape[0]
    splitter = ShuffleSplit(n_splits=1, test_size=prop_not_train)
    train_i, test_i = next(splitter.split(feats))
    if isinstance(train_size, int):
        train_i = train_i[:train_size] #reduce the training set size
    train_package = get_subset(raw_data, train_i)

    pretest_data = get_subset(raw_data, test_i)
    
    #further split test set into test/validation
    prop_val = num_val / test_i.shape[0]
    splitter = ShuffleSplit(n_splits=1, test_size=prop_val) 
    test_i, val_i = next(splitter.split(pretest_data['feats']))
 
    test_package = get_subset(pretest_data, test_i)
    val_package = get_subset(pretest_data, val_i)
    
    #designate the anchor/alignment/reference/whatever set in the training data
    train_gt_cen = train_package['gt_cen_in'][0]
    train_gt_times = train_package['gt_obs_times'][0]
    train_uncen = np.where(last_time[train_i] >= args['horizon']-1)[0]
    print(train_uncen.shape, args['horizon'])
    anc_pts = np.random.choice(train_uncen.shape[0], size=(anc_size - num_val,), replace=False)
    print(anc_pts.shape)
    anc_in = np.zeros((train_i.shape[0],))
    anc_in[train_uncen[anc_pts]] = 1
    train_package['anc_in'] = torch.Tensor(anc_in)
    
    print('dataset split sizes:', train_i.shape, test_i.shape, val_i.shape) 
    return train_package, test_package, val_package
    
    
def split_data_mimic(raw_data, split_seed, args, last_time):  
    np.random.seed(123456789 + split_seed)
    random.seed(123456789 + split_seed)
    
    feats = raw_data['feats']
    cen_lab = raw_data['gt_cen_in'][0]
    train_size = args['train_size']
    anc_size = args['anc_size']
    num_val = 500 

    #split into training/not training
    prop_not_train = (num_val + 2000) / raw_data['feats'].shape[0]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=prop_not_train)
    train_i, test_i = next(splitter.split(feats, cen_lab))
    if isinstance(train_size, int):
        train_i = train_i[:train_size] #reduce the training set size
    train_package = get_subset(raw_data, train_i)

    pretest_data = get_subset(raw_data, test_i)
    
    #further split test set into test/validation
    prop_val = num_val / test_i.shape[0]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=prop_val) 
    test_i, val_i = next(splitter.split(pretest_data['feats'], cen_lab[test_i]))
 
    test_package = get_subset(pretest_data, test_i)
    val_package = get_subset(pretest_data, val_i)
    
    #designate the anchor/alignment/reference/whatever set in the training data
    train_gt_cen = train_package['gt_cen_in'][0]
    train_gt_times = train_package['gt_obs_times'][0]
    train_uncen = np.where(last_time[train_i] >= args['horizon']-1)[0]
    anc_pts = np.random.choice(train_uncen.shape[0], size=(anc_size - num_val,), replace=False)
    print(anc_pts.shape)
    anc_in = np.zeros((train_i.shape[0],))
    anc_in[train_uncen[anc_pts]] = 1
    train_package['anc_in'] = torch.Tensor(anc_in)
    
    print('dataset split sizes:', train_i.shape, test_i.shape, val_i.shape) 
    return train_package, test_package, val_package
    
    
################################################################################################################
'''
get and preprocess dataset by name
'''
def get_dataset(dataset_name, args, split_seed=0):
    print(dataset_name)
    
    if 'synth' in dataset_name:
        dataset, coef, out_var, ncoef, neg_thresh, last_time = get_synth(args)
        return split_data(dataset, split_seed, args, last_time), coef, out_var, ncoef, neg_thresh
    
    elif 'mimic' in dataset_name:
        dataset, coef, out_var, ncoef, neg_thresh, last_time = get_mimic(args)
        return split_data_mimic(dataset, split_seed, args, last_time), coef, out_var, ncoef, neg_thresh


################################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')
    get_mimic('arf', {}, {}, {})
