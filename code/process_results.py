import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pickle
import copy
import seaborn as sns

from exp_settings import *

approach_names = {'baseline_plain_avgnaive': 'Naive Average', 'baseline_plain_avgvote': 'Voting Average', \
                   'baseline_separate': 'Independent', \
                   'proposed3_obs': 'Repeated Samples', \
                   'proposed3': 'Proposed -Error', \
                   'proposed3_100000': 'Proposed -Error 20x Train', \
                   'proposed3_err': 'Proposed', \
                   'proposed3_noneg': 'Proposed -Negative', \
                   'proposed3_margerr': 'Proposed -Recalibration', \
                   'baseline_plain_gt': 'Ground Truth'}
metric_names = {'c index': 'C Index', 'bs': 'IMSE', \
                'c index all time points': 'Consistency', \
                'predicted mean tte minus gt tte': 'Median Predicted TTE - Ground Truth TTE', \
                'predicted median tte minus gt tte': 'Median Predicted TTE - Ground Truth TTE', \
                'censored median accuracy': 'Accuracy on Censored', \
                'auroc': 'AUROC', 'ddc': 'DDC'}
   
#https://matplotlib.org/stable/api/markers_api.html     
markers = ['o', 'v', '^', '>', '<', 's', 'P', 'X', 'D', '*', 'p']
marker_map = {'baseline_plain_avgnaive': 'o', 'baseline_plain_avgvote': 'v', \
             'baseline_separate': '^', 'proposed3_obs': '>', 'proposed3': '<', \
             'proposed3_err': 'X', \
             'proposed3_noneg': 'D', 'proposed3_margerr': '*', \
             'proposed3_100000': 'p'}
             
#https://matplotlib.org/stable/gallery/color/named_colors.html             
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'gold']                           
color_map = {'baseline_plain_avgnaive': 'C0', 'baseline_plain_avgvote': 'C1', \
             'baseline_separate': 'C2', 'proposed3_obs': 'C3', 'proposed3': 'C4', \
             'proposed3_err': 'C6', \
             'proposed3_noneg': 'C8', 'proposed3_margerr': 'C9', \
             'proposed3_100000': 'gold'}
color_map2 = {'baseline_plain_avgnaive': 'skyblue', 'baseline_plain_avgvote': 'sandybrown', \
             'baseline_plain_avgcen': 'lightgreen', \
             'baseline_separate': 'lightgreen', 'proposed3_obs': 'lightcoral', 'proposed3': 'plum', \
             'proposed3_ce': 'C7', 'proposed3_exp': 'C5', 'proposed3_err': 'pink', \
             'proposed3_noneg': 'khaki', 'proposed3_margerr': 'lightsteelblue', \
             'proposed3_100000': 'gold'}
                                
results_dir = 'results/' 


'''
get the name of the results file as formatted in util.py
'''
def get_file_name(date, approach, data_params, dataset, neg=False):
    res_name = date + '_' + approach \
               + '_' + str(data_params['cen_prop']) \
               + '_' + str(data_params['train_size']) \
               + '_' + str(data_params['anc_size']) \
               + '_' + str(data_params['noise']) \
               + '_' + str(data_params['offset'])
    
    final_res_name = results_dir + dataset + '/' + res_name + ".pkl"
    return final_res_name
    
    
'''
eval keys: see metrics in main block
the function is called for each seed separately
outer list - approaches
inner list - settings
'''  
def get_res(file_names, metric):
    results = []

    for i in range(len(file_names)):
        results.append([])
        for file_name in file_names[i]:
            file_handle = open(file_name.replace('_step', ""), 'rb')
            res = pickle.load(file_handle)
            file_handle.close()
            
            step_approach = False
            if 'step' in file_name:
                step_approach = True
            
            if step_approach and metric == 'c index':
                results[i].append(res['c index wrt tte'])
            elif step_approach and metric == 'bs':
                results[i].append(res[metric]['step'])
            elif metric == 'bs':
                results[i].append(res[metric]['preds'])
            else:
                results[i].append(res[metric])
    
    return {metric: results}
    

'''
for each metric, get the median and iqr
'''
def plot_helper(results, apr_i):
    metrics = list(results[0].keys())
    aggregated = {}
    error_lower = {}
    error_upper = {}
    
    for metric in metrics:
        aggregated[metric] = []
        error_lower[metric] = []
        error_upper[metric] = []
    
    #loop over seeds
    for i in range(len(results)):
        for metric in metrics:
            res_i = results[i][metric][apr_i]
            if 'c index' in metric: 
                aggregated[metric].append(res_i)
            elif 'censored median accuracy' in metric:
                aggregated[metric].append([res_i[j][0] for j in range(len(res_i))])
            else:
                medians = [np.percentile(res_i[j], 50) for j in range(len(res_i))]
                iqr_lower = [np.percentile(res_i[j], 25) for j in range(len(res_i))]
                iqr_upper = [np.percentile(res_i[j], 75) for j in range(len(res_i))]
                aggregated[metric].append(medians)
                error_lower[metric].append(iqr_lower)
                error_upper[metric].append(iqr_upper)
       
    return aggregated, error_lower, error_upper
    

'''
get line to plot
'''
def get_plot_line(i, agg_res, metric):
    #print(i, len(agg_res))
    avg = np.median(agg_res[i][metric], axis=0) 
    lower = (avg - np.percentile(agg_res[i][metric], 25, axis=0)).reshape(1, -1)
    upper = (np.percentile(agg_res[i][metric], 75, axis=0) - avg).reshape(1, -1)
    error_bars = np.concatenate([lower, upper], axis=0)
    
    raw_data = []
    for j in range(np.array(agg_res[i][metric]).T.shape[0]):
        raw_data.append(np.array(agg_res[i][metric]).T[j, :])
    
    return avg, error_bars, raw_data
    

'''
print results in a somewhat visually pleasing way
'''
def print_res_tab(res, error_bars, approach):
    res_tab = np.concatenate([res.reshape(-1, 1), error_bars.T], axis=1)
    res_tab[:, 1] = res_tab[:, 0] - res_tab[:, 1]
    res_tab[:, 2] = res_tab[:, 0] + res_tab[:, 2]
    print(approach) 
    print(res_tab)
   
    
'''
do the actual plotting
'''
def plot_res2(approaches, results, dataset, xaxis, xlab, metric, num_plot, plot_num, pcen):
    num_apr = len(results[0][metric])
    
    agg_res = []
    error_lower = []
    error_upper = [] 
    for i in range(num_apr):
        res = plot_helper(results, i)
        agg_res.append(res[0])
        error_lower.append(res[1])
        error_upper.append(res[2])
        
    plt.subplot(1, num_plot, plot_num+1)  
    for i in range(num_apr):   
        med, error_bars = get_plot_line(i, agg_res, metric)
        m, c = marker_map[approaches[i]], color_map[approaches[i]]
        appr = approach_names[approaches[i]]
        if pcen and appr=='Proposed -Error':
            appr = 'Proposed'
        line = 'solid'
        plot_x = xaxis
        if pcen:
            plot_x = np.array(xaxis) + (-2+i)*0.01
        temp_plot = plt.errorbar(plot_x, med, yerr=error_bars, label=appr, marker=m, color=c, \
                    capsize=3, linestyle=line, markersize=4)
        temp_plot[-1][0].set_linestyle(line)
        print_res_tab(med, error_bars, approaches[i])
           
        if 'tte minus gt tte' in metric:
            plt.plot(xaxis, np.zeros((len(xaxis),)), linestyle='--', color='k')
        plt.xlabel(xlab)
        plt.ylabel(metric_names[metric])
        if plot_num == 0:
            plt.legend() 
            
            
'''
do the actual plotting
'''
def plot_res(approaches, results, dataset, xaxis, xlab, metric, num_plot, plot_num, pcen):
    num_apr = len(results[0][metric])
    
    agg_res = []
    error_lower = []
    error_upper = []
    for i in range(num_apr):
        res = plot_helper(results, i)
        agg_res.append(res[0])
        error_lower.append(res[1])
        error_upper.append(res[2])
    
    plt.subplot(1, num_plot, plot_num+1)
    if 'tte minus gt tte' in metric and dataset != 'mimic':
        plt.plot(np.array(xaxis)-0.1, np.zeros((len(xaxis),)), linestyle='--', color='k')
    
    for i in range(num_apr):   
        med, error_bars, raw_res = get_plot_line(i, agg_res, metric)
        m, c = marker_map[approaches[i]], color_map[approaches[i]]
        
        appr = approach_names[approaches[i]]
        if pcen and appr=='Proposed -Error':
            appr = 'Proposed'
        plot_x = xaxis
        if pcen:
            plot_x = np.array(xaxis) + (-2+i)*0.015 - 0.1
        elif -20 in xaxis: 
            plot_x = np.array(xaxis) + (-1+i)
        else: 
            plot_x = np.array(xaxis) + (-1+i)*0.01
        #bar plot version
        if metric == 'c index' and dataset != 'mimic':
            width = 0.012
            if -20 in xaxis:
                width = 0.9
            if dataset == 'mimic':
                width = 40
            max_val = np.array([np.max(res) for res in raw_res]) - med
            min_val = med - np.array([np.min(res) for res in raw_res])
            error_bars = np.concatenate([min_val.reshape(1, -1), max_val.reshape(1, -1)], axis=0)
            plt.bar(plot_x, med, yerr=error_bars, label=appr, color=c, capsize=2, width=width, ecolor='gray')
            plt.ylim(bottom=0.4)
            if dataset == 'mimic':
                plt.ylim(bottom=0.3)
        #violin plot version
        if metric != 'c index' or dataset == 'mimic':
            print(c)
            vwidth = 0.03
            if pcen and metric == 'c index':
                vwidth = 0.01
            if dataset == 'mimic':
                vwidth = 45
            elif -20 in xaxis:
                vwidth = 1
            vparts = plt.violinplot(raw_res, plot_x, widths=vwidth, quantiles=[[0.25, 0.75] for v in plot_x])
            for pc in vparts['bodies']:
                pc.set_facecolor(c)
                pc.set_edgecolor(c)
                pc.set_alpha(0.3)
            for pc in ('cbars', 'cmins', 'cmaxes', 'cquantiles'):
                if 1==1:#metric != 'c index':
                    vparts[pc].set_edgecolor(color_map2[approaches[i]])
                else:
                    vparts[pc].set_edgecolor('silver')
            if not (pcen and metric == 'c index'):
                plt.plot(plot_x, med, label=appr, marker=m, color=c, linestyle='solid', markersize=4)
            
        print_res_tab(med, error_bars, approaches[i])
           
        plt.xlabel(xlab)
        plt.ylabel(metric_names[metric])
        if plot_num == 0:
            plt.legend()      
     

'''
do the actual plotting for the neg experiment
'''
def plot_neg(approaches, results, dataset, xaxis, xlab, yaxis, ylab, metric, plot_num):
    num_apr = len(results[0][metric])
    
    agg_res = []
    error_lower = []
    error_upper = []
    for i in range(num_apr):
        res = plot_helper(results, i)
        agg_res.append(res[0])
        error_lower.append(res[1])
        error_upper.append(res[2])
    
    plt.subplot(1, 4, plot_num+1)
    for i in range(num_apr): 
        for j in range(len(agg_res[i][metric])):
            x_axis = []  
            agg_res2 = {}
            for a in range(5):
                for b in range(5):
                    if (a+1) + (b+1) in x_axis:
                        agg_res2[(a+1) + (b+1)].append(agg_res[i][metric][j][(a*5)+b])
                    else:
                        x_axis.append((a+1) + (b+1))
                        agg_res2[(a+1) + (b+1)] = [agg_res[i][metric][j][(a*5)+b]]
            agg_res2 = [np.mean(np.array(agg_res2[k])) for k in x_axis]
            agg_res[i][metric][j] = agg_res2
        med, error_bars = get_plot_line(i, agg_res, metric)
        m, c = marker_map[approaches[i]], color_map[approaches[i]]
        plt.errorbar(np.array(x_axis)*0.05, med, yerr=error_bars, label=approach_names[approaches[i]], marker=m, color=c)
        print_res_tab(med, error_bars, approaches[i]) 
    if 'tte minus gt tte' in metric and dataset != 'mimic':
        plt.plot(xaxis, np.zeros((len(xaxis),)), linestyle='--', color='k')
    plt.xlabel('Noise Rate')
    plt.ylabel(metric_names[metric])
    
    


###################################################################################################
'''
main block
'''
if __name__ == '__main__':
    dataset_name = 'synth'
    exp_name = 'size'
    date = '231210'
    seeds = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    approaches1 = ['baseline_plain_avgnaive', 'baseline_plain_avgvote', 'baseline_separate', \
                   'proposed3_obs', 'proposed3']#, 'proposed3_100000']
    approaches2 = ['proposed3', 'proposed3_margerr', 'proposed3_err']
    approaches3 = ['proposed3_noneg', 'proposed3_err']
    approaches_mimic = ['baseline_plain_avgnaive', 'baseline_plain_avgvote', \
                   'baseline_separate', 'proposed3_obs', 'proposed3_err']
    
    if exp_name == 'pcen': #see how performance changes with proportion partially censored (baselines)
        approaches = approaches1
        exp_settings = all_exp_settings[dataset_name][0:9]
        xaxis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        xlab = 'Proportion Censored at Training'
    elif exp_name == 'offset': #see how performance changes with differnt noise means
        date = '231211' 
        approaches = approaches2
        exp_settings = all_exp_settings[dataset_name][10:21]
        xaxis = [-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20]
        xlab = 'Noise Mean'
    elif exp_name == 'neg': #see how performance changes with difference noise rates wrt occurrence
        date = '231211' 
        approaches = approaches3
        exp_settings = all_exp_settings[dataset_name][21:31]
        xaxis = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        xlab = 'Noise Rate WRT Event Occurrence'
    elif exp_name == 'size': 
        dataset_name = 'mimic' #this was done on mimic data
        date = '240911'
        approaches = ['proposed3_err']
        exp_settings = all_exp_settings[dataset_name][1:11]
        xaxis = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        xlab = 'Number of Reference Samples in Training Set'
    elif exp_name == 'mimic': #mimic data
        dataset_name = 'mimic'
        date = '240131'
        approaches = approaches_mimic
        exp_settings = all_exp_settings[dataset_name][:1]
        xaxis = [1]
        xlab = 'Placeholder'
    
    metrics = ['predicted median tte minus gt tte', 'c index', 'bs']
    if exp_name == 'neg':
        metrics = ['predicted median tte minus gt tte', 'c index', 'bs', 'auroc']
    if dataset_name != 'synth':
        metrics[-1] = 'ddc'
        metrics.append('auroc')
    
    #plot metrics for all approaches
    plt.figure(figsize=(12, 4))
    if len(metrics) > 3:
        plt.figure(figsize=(18,4))
    for l in range(len(metrics)):
        metric = metrics[l]
        print(metric)
        all_res = []
        for i in range(len(seeds)):
            test_files_all = []
            date_i = date + '_' + seeds[i]
            for j in range(len(approaches)):
                #date, approach, data_params, resutls_dir, dataset
                test_files = []
                for k in range(len(exp_settings)):
                    setting = copy.deepcopy(exp_settings[k])
                    method = copy.deepcopy(approaches[j])
                    if approaches[j] == 'proposed3_100000':
                        setting['train_size'] = 100000
                        method = 'proposed3'
                    test_files.append(get_file_name(date_i, method, setting, dataset_name, exp_name=='neg'))
                test_files_all.append(test_files)
            res = get_res(test_files_all, metric)
            all_res.append(res) 
        plot_res(approaches, all_res, dataset_name, xaxis, xlab, metric, len(metrics), l, exp_name=='pcen')
    plt.subplots_adjust(bottom=0.15, right=0.98, left=0.08, top=0.95)
    save_name = exp_name
    plt.savefig('plots/'+ dataset_name + '_' + save_name + '.png', dpi=300)
    print('\n')
    
    
    
