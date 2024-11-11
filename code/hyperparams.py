'''
record of all hyperparameters

the dictionaries synth_hyperparams and mimic_hyperparams can be ignored since they are not used
instead, synth_ranges and mimic_ranges are used to search over a space of hyperparameters
'''

###################################################################################################
'''
synthetic data 
'''

synth_hyperparams = \
{
    'baseline_plain': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100}, \
    'baseline_separate': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'layer_div': 2, 'layer_s': 100}, \
    'proposed3': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'layer_div': 2, 'layer_s': 100, 'burn_in': 100, 'exp_weight': 0.1, 'ent_weight': 0.1}, \
}

synth_ranges = \
{
    'baseline_plain': {'l_rate': [0.001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s': [100, 100], \
                       'cen_weight': [1, 1]}, \
    'baseline_separate': {'l_rate': [0.001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s':[100, 100], \
                          'cen_weight': [1, 1]}, \
    'proposed3': {'l_rate': [0.001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [2, 2], 'layer_s':[100, 100], \
                  'cen_weight': [1, 1], 'burn_in': [150,150], 'exp_weight': [0.0, 0.0], 'ent_weight': [1, 1]}, \
}


'''
mimic iii with sepsis
'''
mimic_hyperparams = \
{
    'baseline_plain': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'n_layer': 2, 'layer_s': 100}, \
    'baseline_separate': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'layer_div': 2, 'layer_s': 100}, \
    'proposed3': {'l_rate': 0.001, 'l2': 0.01, 'batch': 5, 'layer_div': 2, 'layer_s': 100, 'burn_in': 100, 'exp_weight': 0.1, 'ent_weight': 0.1}, \
}

mimic_ranges = \
{
    'baseline_plain': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [3, 3], 'layer_s': [100, 100], \
                       'cen_weight': [0.01, 0.1]}, \
    'baseline_separate': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1],'batch': [5, 5], 'n_layer': [3, 3], 'layer_s': [100, 100], \
                       'cen_weight': [0.01, 0.1]}, \
    'proposed3': {'l_rate': [0.0001, 0.01], 'l2': [0.001, 0.1], 'batch': [5, 5], 'n_layer': [3, 3], 'layer_s': [100, 100], \
                      'cen_weight': [0.01, 0.1], 'burn_in': [150,150], 'exp_weight': [0.01, 0.4], 'ent_weight': [1, 1]}, \
}


###################################################################################################
'''
putting everything together
'''

all_hyperparams = \
{
    'synth': synth_hyperparams, \
    'mimic': mimic_hyperparams, \
}

hp_ranges = \
{
    'synth': synth_ranges, \
    'mimic': mimic_ranges, \
}

###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')
