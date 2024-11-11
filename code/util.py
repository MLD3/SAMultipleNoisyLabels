'''
functions related to training and evaluation live here
general purpose, all use functions
'''

import copy
import random
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

import torch
import torch.optim as optim
import torch.nn as nn

import base_feed_for
import base_separate
import proposed3

################################################################################################
'''
put things on gpu
'''
def to_gpu(for_gpu):
    dev = 'cpu'
    gpus = [1 ,2, 3, 4, 5, 6, 7, 0]
   
    for i in range(len(gpus)):
        if torch.cuda.is_available():
            dev = 'cuda:' + str(gpus[i])
        device = torch.device(dev)

        if not isinstance(for_gpu, list):
            for_gpu = for_gpu.to(device)
            if for_gpu.device.type != 'cpu':
                return for_gpu
        else:
            gpu_items = []
            for item in for_gpu:
                gpu_items.append(item.to(device))
            if gpu_items[0].device.type != 'cpu':
                return gpu_items
    
    
################################################################################################
'''
naive ways to aggregate multiple labels/time to events
'''

def aggregate_min(times, cens):
    num_ex = times.shape[1]
    num_lab = times.shape[0]

    times_final = np.zeros((num_ex,))
    cens_final = np.zeros((num_ex,))
    for i in range(num_ex):
        time_i = np.argmin(times[:, i])
        times_final[i] = times[time_i, i]
        cens_final[i] = cens[time_i, i]
        
    return times_final, cens_final
    
def aggregate_max(times, cens):
    num_ex = times.shape[1]
    num_lab = times.shape[0]

    times_final = np.zeros((num_ex,))
    cens_final = np.zeros((num_ex,))
    
    for i in range(num_ex):
        censored_i = np.where(cens[:, i] == 1)[0]
        if censored_i.shape[0] == 0:
            times_final[i] = np.max(times[:, i])
            cens_final[i] = 0
            continue
            
        prop_cen = censored_i.shape[0] / num_lab
        censored = np.random.binomial(1, prop_cen)
        if censored:
            times_final[i] = np.max(times[:, i])
            cens_final[i] = 1
        else:
            uncensored = np.where(cens[:, i] == 0)[0]
            times_final[i] = np.max(times[uncensored, i])
            cens_final[i] = 0
            
    return times_final, cens_final
    
def aggregate_avg(times, cens, naive=True):
    num_ex = times.shape[1]
    num_lab = times.shape[0]

    times_final = np.zeros((num_ex,))
    cens_final = np.zeros((num_ex,))
    for i in range(num_ex):
        censored_i = np.where(cens[:, i] == 1)[0]
        if censored_i.shape[0] == 0:
            times_final[i] = np.mean(times[:, i])
            cens_final[i] = 0
            continue
            
        prop_cen = censored_i.shape[0] / num_lab
        censored = 0 if prop_cen < 1 else 1#naive averaging
        if not naive:
            censored = np.random.binomial(1, prop_cen) #probabilistic voting averaging
        if censored:
            times_final[i] = np.max(times[:, i])
            cens_final[i] = 1
        else:
            uncensored = np.where(cens[:, i] == 0)[0]
            times_for_avg = copy.deepcopy(times[:, i])
            if naive == 'Part':
                uncensored = np.arange(num_lab)
                times_for_avg[np.where(cens[:, i] == 1)[0]] += 1
            times_final[i] = np.mean(times_for_avg[uncensored])
            cens_final[i] = 0
        
    return times_final, cens_final
    
def aggregate_rand(times, cens):
    num_ex = times.shape[1]
    num_lab = times.shape[0]

    times_final = np.zeros((num_ex,))
    cens_final = np.zeros((num_ex,))
    
    for i in range(num_ex):
        label = np.random.randint(0, num_lab)
        times_final[i] = times[label, i]
        cens_final[i] = cens[label, i]
    
    return times_final, cens_final
    
'''
main function, combines the above
'''
def aggregate(times_orig, cens_orig, method, naive=True):
    times = [times_orig[i].detach().cpu().numpy() for i in range(len(times_orig))]
    cens = [cens_orig[i].detach().cpu().numpy() for i in range(len(cens_orig))]
    times = np.array(times)
    cens = np.array(cens)
    
    if method == 'min':
        new_times, new_cens = aggregate_min(times, cens)
        
    elif method == 'max':
        new_times, new_cens = aggregate_max(times, cens)
        
    elif method == 'avg':
        new_times, new_cens = aggregate_avg(times, cens, naive)
        
    elif method == 'rand':
        new_times, new_cens = aggregate_rand(times, cens)
        
    return [torch.Tensor(new_times)], [torch.Tensor(new_cens)]


################################################################################################
'''
train a model
'''
def train_model(model, loss_fx, hyperparams, dataset, data_params, approach):
    #unpack
    train_data, val_data = dataset['train'], dataset['val']
    train_cov, train_times, train_cen_in = train_data['feats'], train_data['obs_times'], train_data['cen_in']
    train_times_gt, train_cen_in_gt = train_data['gt_obs_times'], train_data['gt_cen_in']
    train_anc = train_data['anc_in']
    val_cov, val_times, val_cen_in = val_data['feats'], val_data['gt_obs_times'], val_data['gt_cen_in']
    
    #setup
    l_rate, l2_const, num_batch = hyperparams['l_rate'], hyperparams['l2'], hyperparams['batch']
    mod_params = model.get_parameters()
    optimizer = optim.Adam(mod_params, lr=l_rate, weight_decay=l2_const) 
    min_epochs = data_params['min_ep']
    max_epochs = data_params['max_ep']
    horizon = data_params['horizon']
    patience = 20

    #use gpu >_<
    train_cov, val_cov, model = to_gpu([train_cov, val_cov, model])
    
    stop_crit = 'mean_tte_minus_gt'
    stop_loss = False
    
    #initial "evaluation" ;)
    val_eval = -100000
    eval_diff = 100000
    eval_prev = 100000
    if stop_crit == 'c index':
        eval_prev = -100000
    eval_tol = 1e-4
    val_avg = 0
    
    #train model 
    i = 1
    prev_mod = copy.deepcopy(model)
    while (eval_diff > eval_tol or i < min_epochs) and i < max_epochs:
        train_loss = 0     
        splitter = KFold(num_batch, shuffle=True)
        batch_split = splitter.split(train_cov) 
        for j in range(num_batch):
            _, batch_ind = next(batch_split)
            extra_args = {'epoch': i}
            train_out = model(train_cov[batch_ind, :], return_extra=True, extra_args=extra_args)
            batch_times = [train_times[k][batch_ind] for k in range(len(train_times))]
            batch_cen_in = [train_cen_in[k][batch_ind] for k in range(len(train_cen_in))]
            batch_gt_times = train_times_gt[0][batch_ind]
            batch_gt_cen = train_cen_in_gt[0][batch_ind]
            batch_anc = train_anc[batch_ind]
            loss_args = {'cen_in': batch_cen_in, 'epoch': i, 'gt_times': batch_gt_times, \
                         'gt_cen': batch_gt_cen, 'anc_in': batch_anc}
            batch_loss = loss_fx(train_out, batch_times, loss_args)
            
            if batch_loss != 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                train_loss += (batch_loss.detach() / num_batch)
        
        if not stop_loss:
            val_out = model(val_cov, return_extra=False)        
            val_eval = evaluate(val_out, val_times, val_cen_in, horizon)[stop_crit]
            val_avg += abs(val_eval) / patience 
            
        #evaluate every 20 epochs on validation data 
        if i % patience == 0 and i != 0:    
            eval_diff = -(val_avg - eval_prev)
            if stop_crit == 'c index':
                eval_diff = (val_avg - eval_prev)
            if eval_diff > 0:
                prev_mod = copy.deepcopy(model)
                eval_prev = copy.deepcopy(val_avg)
            val_out2 = model(val_cov, return_extra=False)       
            val_eval2 = abs(evaluate(val_out2, val_times, val_cen_in, horizon)[stop_crit]) 
            
            print('new training evaluation')
            print(i, val_avg, eval_diff)
            print('training loss: ', train_loss) 
            val_avg = 0  
        i += 1

    print('done training', eval_prev)  
    if stop_crit == 'c index':
        return prev_mod, -eval_prev, i 
    return prev_mod, eval_prev, i


################################################################################################
'''
calculate a survival curve
'''
def get_surv_curve(preds):
    curve = np.zeros(preds.shape)
    num_time_steps = preds.shape[1]
    
    for i in range(num_time_steps):
        curve[:, i] = 1 - np.sum(preds[:, :i+1], axis=1)
        
    return curve
    
    
'''
calculate a survival curve but as a step function
'''
def get_step_curve(preds):
    curve = np.zeros(preds.shape)
    num_time_steps = preds.shape[1]
    
    #get conditional probs
    denom = 1-preds[:, -1].reshape(-1, 1)
    denom[denom == 0] = 1
    preds_cond = (preds[:, :-1] / denom) 
        
    #expected value of obs distr preds - regular version and noisy version
    nums = np.arange(preds_cond.shape[1]).reshape(1, -1)
    exp_tte = np.sum(preds_cond * nums, axis=1)
    
    #make the curve
    for i in range(num_time_steps):
        had_event = np.where(exp_tte <= i)[0]
        fill_val = preds[had_event, -1]
        curve[:, i] = 1 
        curve[had_event, i] = fill_val
        
    return curve
    
    
'''
calculate a survival curve, ground truth using oracle information
if need to update this to account for negatives - make tte and censorship an input to this function to check for 'negative status'
    and then make survival curve a flat line at y=1 for any negatives
'''
def get_gt_curve(preds, feats, coef, var, horizon, ncoef, neg_thresh):
    curve = np.zeros(preds.shape)
    num_time_steps = preds.shape[1]
    
    outcome = coef * feats
    outcome = np.sum(outcome, axis=1)
    outcome = (outcome - np.min(outcome)) / (np.max(outcome) - np.min(outcome))
    outcome = np.floor(outcome*(horizon-1)).astype(int)
    
    neg = np.sum(ncoef * feats, axis=1)
    neg_ex = np.where(neg < neg_thresh)[0]
    if isinstance(ncoef, int):
        neg_ex = []
    
    num_samples = 10000
    probs = np.zeros((feats.shape[0], num_time_steps))
    
    for i in range(len(outcome)):
        if i in neg_ex:
            continue
        sample_i = np.random.normal(outcome[i], var, size=(num_samples,)).astype(int)
        sample_i[sample_i >= num_time_steps] = num_time_steps - 1
        sample_i[sample_i < 0] = 0
        for j in range(num_time_steps):
            probs[i, j] = np.where(sample_i == j)[0].shape[0] / num_samples
    
    for i in range(num_time_steps):
        curve[:, i] = 1 - np.sum(probs[:, :i], axis=1)
        
    return curve
    
    
'''
calculate c index, evaluate at time of event
'''
def calculate_c1(outcomes, times, cens=None, labeler=0):
    cens = cens[labeler]
    times = times[labeler]
    horizon = torch.max(times)
    
    numer, denom = 0, 0
    num_steps = outcomes.shape[1]
    surv_curve = get_surv_curve(outcomes)

    for i in range(len(outcomes) - 1):
        if cens is not None and cens[i] == 1:
            continue
        time = int(times[i])
        risk = surv_curve[i, time]
        compare_i = (np.where(times[i+1:] > time)[0] + i + 1).astype(int)
        if cens is not None:
            compare_c1 = (np.where(times[i+1:] == time)[0] + i + 1).astype(int)
            compare_c2 = (np.where(cens[i+1:] == 1)[0] + i + 1).astype(int)
            compare_c = np.intersect1d(compare_c1, compare_c2)
            compare_i = np.union1d(compare_i, compare_c)
        
        compare_r = surv_curve[compare_i, :][:, time]
        denom += compare_i.shape[0]
        lower_risk = np.where(compare_r > risk)[0].shape[0] #higher probability of survival means lower risk
        same_risk = np.where(compare_r == risk)[0].shape[0]
        numer += lower_risk + 0.5*same_risk
    return numer / denom
    
    
'''
calculate c index, evaluate at each relevant time point
'''
def calculate_c(outcomes, times, cens=None, labeler=0):
    cens = cens[labeler]
    times = times[labeler]
    
    numer, denom = 0, 0
    num_steps = outcomes.shape[1]
    surv_curve = get_surv_curve(outcomes)
    
    times2 = copy.deepcopy(times)
    times2[cens == 1] += 1
    time_tile = np.tile(times2, (outcomes.shape[0], 1))
    cen_tile = np.tile(cens, (outcomes.shape[0], 1))
    time_diff = time_tile - time_tile.T

    for i in range(num_steps):
        risk_diff1 = np.tile(surv_curve[:, i], (outcomes.shape[0], 1))
        risk_diff = risk_diff1 - risk_diff1.T
        #get anyone who had event at or before time i
        indicator = np.logical_and(time_diff < 0, time_tile <= i)  
        
        indicator = np.logical_and(indicator, cen_tile == 0)
        #pair them with the observed time after time i (accounts for censoring)
        indicator = np.logical_and(indicator, time_tile.T > i)
        
        compare = risk_diff * indicator
        compare[compare > 0] = 0
        compare[compare == 0] = 0.5
        compare[compare < 0] = 1 
        compare[indicator == 0] = 0
        normalizer = np.abs(time_diff)
        normalizer[normalizer == 0] = 1
        numer += np.sum(compare / normalizer)
        denom += np.sum(indicator / normalizer)
    
    return numer / denom
    

def get_tte_stats(preds, event_times, cens, test_time=False, mimic=False):
    is_offset = False
    if isinstance(preds, list):
        is_offset = True
        preds, bias = preds[0], preds[1]
        bias_size = bias.shape[1] // 2
        bias_range = np.concatenate([np.arange(-bias_size, 0), np.arange(1, bias_size+1)])
        
    denom = 1-preds[:, -1].reshape(-1, 1)
    denom[denom == 0] = 1
    preds_cond = preds[:, :-1] / denom
    times = event_times[0]
    
    surv_curve = get_surv_curve(preds)
    num_steps = preds.shape[1]
    
    prob_at_tte = []
    med_minus_tte = []
    uncen = np.where(cens[0] == 0)[0]
    mean_minus_tte = preds_cond * np.arange(preds_cond.shape[1]).reshape(1, -1)
    mean_minus_tte = np.sum(mean_minus_tte[uncen], axis=1) - times[uncen].detach().cpu().numpy()  
    cen_med_acc = 0
    num_cen = 0
    
    for i in range(preds.shape[0]):
        tte = int(times[i])
        if cens[0][i] == 0 or mimic:
            prob_at_tte.append(surv_curve[i, tte])
        had_event = np.where(surv_curve[i, :]<=0.5)[0]
        if had_event.shape[0] < 1:
            had_event = [preds_cond.shape[1]]
        
        median = np.min(had_event)
        
        if cens[0][i] == 1: 
            num_cen += 1
            if median > tte:
                cen_med_acc += 1
            continue         
        
        med_minus_tte.append(median - tte)
   
    stats = {}
    stats['survival probability at tte'] = prob_at_tte
    stats['predicted median tte minus gt tte'] = med_minus_tte
    stats['predicted mean tte minus gt tte'] = mean_minus_tte
    stats['censored median accuracy'] = cen_med_acc / max(num_cen, 1), num_cen
    stats['ddc'] = get_ddlc(np.array(prob_at_tte))
    return stats
    
    
'''
get auroc
'''
def get_auroc(event_times, preds, cens, horizon, labeler):
    if isinstance(preds, list):
        preds = preds[0]
    uncen = np.where(cens[labeler] == 0)[0]
    uncen2 = np.where(event_times[labeler] == horizon-1)[0]
    uncen = np.union1d(uncen, uncen2)
    
    class_preds = 1 - preds[uncen, -1] 
    labs = np.ones(class_preds.shape)
    neg = np.where(np.logical_and(event_times[labeler][uncen]==horizon-1, cens[labeler][uncen]==1))[0]
    labs[neg] = 0
    
    auroc = 1
    if np.unique(labs).shape[0] > 1:
        auroc = roc_auc_score(labs.astype(int), class_preds)
        
    return auroc
  
'''
brier score because reasons
technically it's the average brier over all time points
except it's not actually the brier
'''
def calculate_brier(preds, event_times, cens, feats, coef, out_var, horizon, ncoef, neg_thresh):
    pred_curve = get_surv_curve(preds)
    step_curve = get_step_curve(preds)
    gt_curve = get_gt_curve(preds, feats, coef, out_var, horizon, ncoef, neg_thresh)
    
    #brier wrt predicted curve
    pred_brier = np.sum(np.square(pred_curve - gt_curve), axis=1) / gt_curve.shape[1]
    pred_brier = pred_brier[cens[0] == 0]
    
    #brier wrt step fx
    step_brier = np.sum(np.square(step_curve - gt_curve), axis=1) / gt_curve.shape[1]
    step_brier = step_brier[cens[0] == 0]
    
    return {'preds': pred_brier, 'step': step_brier}
    
    
'''
get ddc for calibration of real data
'''
def get_ddlc(probs):
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1]
    bin_props = np.zeros((len(bins) - 1,))
    
    num_probs = len(probs)
    if num_probs == 0:
        return 10000000
    
    for i in range(len(bins) - 1):
        bin_props[i] = np.where(np.logical_and(probs >= bins[i], probs < bins[i+1]))[0].shape[0] / num_probs
    
    ref_probs = np.ones(bin_props.shape) * 0.1
    ddlc = np.sum(bin_props * np.log(bin_props/ref_probs+1e-10))
        
    return ddlc
    
    
'''
plot preds wrt tte
'''
def plot_preds(preds, event_times, cens, feats):
    is_offset = False
    if isinstance(preds, list):
        is_offset = True
        preds, bias = preds[0], preds[1]
        
    num_time = preds.shape[1]
    plot = np.zeros((num_time, num_time))
    
    times = event_times[0]
    times[cens[0] == 1] += 1
    
    for i in range(num_time):
        at_i = np.where(times == i)[0]
        pred_i = np.zeros((num_time,))
        if at_i.shape[0] > 0:
            pred_i = np.average(preds[at_i, :], axis=0)
        plot[:, i] = pred_i
    
    plot_b =  0
    if is_offset:
        plot_b = np.average(bias, axis=0)
          
    return plot, plot_b


'''
overall evaluation, calculate as many things as the heart desires :)
'''
def evaluate(preds, event_times, cens, horizon, test_time=False, feats=None, coef=None, \
             out_var=None, ncoef=None, prop_neg=None, mimic=False):
    num_labelers = len(event_times)
    
    results = {}
    results['med_tte_minus_gt'] = 0
    results['mean_tte_minus_gt'] = 0
    results['median_tte_minus_gt'] = 0
    results['c index'] = 0
    
    if not test_time:
        for i in range(0, num_labelers):
            stats = get_tte_stats(preds, [event_times[i]], [cens[i]])
            results['mean_tte_minus_gt'] += abs(np.percentile(stats['predicted mean tte minus gt tte'], 50)) / num_labelers
            results['median_tte_minus_gt'] += abs(np.percentile(stats['predicted median tte minus gt tte'], 50)) / num_labelers
            results['c index'] += calculate_c1(preds, [event_times[i]], [cens[i]]) / num_labelers
        results['sum'] = abs(results['median_tte_minus_gt']) + 1*(1 - results['c index'])
        
    elif test_time: #ground truth labels so there's only 1 set of labels
        results['c index'] = calculate_c1(preds, event_times, cens)
        results['c index all time points'] = calculate_c(preds, event_times, cens)
        results['auroc'] = get_auroc(event_times, preds, cens, horizon, 0)
        stats = get_tte_stats(preds, event_times, cens, test_time, mimic)
        for key in list(stats.keys()):
            results[key] = stats[key]   
        pred_plot = plot_preds(preds, event_times, cens, feats)
        if isinstance(coef, int):
            results['bs'] = {'preds': np.ones((preds.shape[0],)), 'step': np.ones((preds.shape[0],))}
        else:
            results['bs'] = calculate_brier(preds, event_times, cens, feats, coef, out_var, horizon, ncoef, prop_neg)
    
    results['preds'] = preds    
    return results


################################################################################################
'''
overall wrapper - train/test/validate a model given the dataset, approach, parameters
'''
def get_model(dataset_name, dataset_package, approach, data_params, hyperparams):
    if 'baseline_plain' in approach:
        model, val_res, ep = base_feed_for.get_model(dataset_name, dataset_package, approach, data_params, hyperparams)
    elif 'baseline_separate' in approach:
        model, val_res, ep = base_separate.get_model(dataset_name, dataset_package, approach, data_params, hyperparams)
    elif 'proposed3' in approach:
        model, val_res, ep = proposed3.get_model(dataset_name, dataset_package, approach, data_params, hyperparams)
    
    return model, val_res, ep


'''
hyperparameter tuning
'''
def tune_hyperparams(dataset_package, approach, data_params, hyperparam_ranges, results_dir, date, dataset_name):
    budget = 5
    keys = list(hyperparam_ranges.keys())
    num_hyperparams = len(keys)
    
    test_data = dataset_package[1]
    val_results = np.ones((budget,)) * 1000
    
    best_hyperparams = 1
    best_mod = 1
    num_ep, best_i = 0, -1
    
    for i in range(budget):
        print('hyperparam selection iter ', i)
        hyperparams = {}
        for j in range(num_hyperparams):
            bound = hyperparam_ranges[keys[j]]
            if bound[0] < bound[1]:
                hyperparams[keys[j]] = loguniform.rvs(bound[0], bound[1])
            else:
                hyperparams[keys[j]] = bound[0]
        
        print(hyperparams)
        mod, val_eval, ep = get_model(dataset_name, dataset_package, approach, data_params, hyperparams)
        val_results[i] = val_eval
        
        if val_results[i] == np.min(val_results):
            best_mod = mod
            best_hyperparams = hyperparams
            num_ep, best_i = ep, i
    print('num_epochs, best iter: ', num_ep, best_i, np.min(val_results), val_results)

    test_data = dataset_package[1]
    test_cov, test_times, test_cen_in = test_data['feats'], test_data['gt_obs_times'], test_data['gt_cen_in']
    test_cov = to_gpu([test_cov])[0]
    test_out = best_mod(test_cov)
    test_times = [copy.deepcopy(test_data['gt_times'])] 
    for i in range(len(test_times[0])):
        time = test_times[0][i] 
        if time < data_params['num_steps']-1 or test_cen_in[0][i] == 0:
            test_cen_in[0][i] = 0
        else:
            test_cen_in[0][i] = 1
            test_times[0][i] = data_params['num_steps'] - 1
    
    horizon = data_params['horizon']
    test_eval = evaluate(test_out, test_times, test_cen_in, horizon, test_time=True, \
                feats=test_cov.detach().cpu().numpy(), coef=data_params['coef'], out_var=data_params['out_var'], \
                ncoef=data_params['ncoef'], prop_neg=data_params['prop_neg'], mimic=False)
 
    res_name = date + '_' + approach \
               + '_' + str(data_params['cen_prop']) \
               + '_' + str(data_params['train_size']) \
               + '_' + str(data_params['anc_size']) \
               + '_' + str(data_params['noise']) \
               + '_' + str(data_params['offset'])
    if len(data_params['noise']) == 2:
        res_name = res_name + '_' + str(data_params['same_error']) 
    pickle.dump(test_eval, open(results_dir + dataset_name + '/' + res_name + ".pkl", "wb"))

    return best_mod, best_hyperparams, test_eval


################################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')
