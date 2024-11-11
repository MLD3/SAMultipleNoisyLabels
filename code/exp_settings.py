'''
record of all settings tested in experiments
'''

import numpy as np

synth = \
[
    #pcen
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.0], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.1], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.2], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.3], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.5], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.6], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.7], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.8], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.85], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 0}, \
    #####################################
    #offset
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': -20}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': -16}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': -12}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': -8}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': -4}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 4}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 8}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 12}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 16}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 500, 'offset': 20}, \
    #####################################
    #noise rate
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0.5, \
    'noise': np.array([[0.1, 0.1],[0.1, 0.1],[0.1,0.1]])*0.5, 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0.5, \
    'noise': np.array([[0.1, 0.1],[0.1, 0.1],[0.1,0.1]])*1, 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0.5, \
    'noise': np.array([[0.1, 0.1],[0.1, 0.1],[0.1,0.1]])*1.5, 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0.5, \
    'noise': np.array([[0.1, 0.1],[0.1, 0.1],[0.1,0.1]])*2, 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0.5, \
    'noise': np.array([[0.1, 0.1],[0.1, 0.1],[0.1,0.1]])*2.5, 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0.5, \
    'noise': np.array([[0.1, 0.1],[0.1, 0.1],[0.1,0.1]])*3, 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0.5, \
    'noise': np.array([[0.1, 0.1],[0.1, 0.1],[0.1,0.1]])*3.5, 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0.5, \
    'noise': np.array([[0.1, 0.1],[0.1, 0.1],[0.1,0.1]])*4, 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0.5, \
    'noise': np.array([[0.1, 0.1],[0.1, 0.1],[0.1,0.1]])*4.5, 'anc_size': 500, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 200, 'cen_prop': [0.1, 0.4], 'train_size': 5000, 'num_data': 7250, 'prop_neg': 0.5, \
    'noise': np.array([[0.1, 0.1],[0.1, 0.1],[0.1,0.1]])*5, 'anc_size': 500, 'offset': 0}, \
] 


mimic = \
[
    #just 1 setting :) dataset is 19866 admissions, only num labelers, horizon, train size and anc size matter here
    #full training set is 19866 - validation set - test set (17366 if 500, 2000)
    {'num_labelers': 3, 'horizon': 24, 'cen_prop': [0.0, 0.0], 'train_size': 17366, 'num_data': 19866, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 1000, 'offset': 0}, \
    #####################################
    #vary reference set size
    {'num_labelers': 3, 'horizon': 24, 'cen_prop': [0.0, 0.0], 'train_size': 17366, 'num_data': 19866, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 550, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 24, 'cen_prop': [0.0, 0.0], 'train_size': 17366, 'num_data': 19866, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 600, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 24, 'cen_prop': [0.0, 0.0], 'train_size': 17366, 'num_data': 19866, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 650, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 24, 'cen_prop': [0.0, 0.0], 'train_size': 17366, 'num_data': 19866, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 700, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 24, 'cen_prop': [0.0, 0.0], 'train_size': 17366, 'num_data': 19866, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 750, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 24, 'cen_prop': [0.0, 0.0], 'train_size': 17366, 'num_data': 19866, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 800, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 24, 'cen_prop': [0.0, 0.0], 'train_size': 17366, 'num_data': 19866, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 850, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 24, 'cen_prop': [0.0, 0.0], 'train_size': 17366, 'num_data': 19866, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 900, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 24, 'cen_prop': [0.0, 0.0], 'train_size': 17366, 'num_data': 19866, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 950, 'offset': 0}, \
    {'num_labelers': 3, 'horizon': 24, 'cen_prop': [0.0, 0.0], 'train_size': 17366, 'num_data': 19866, 'prop_neg': 0, \
    'noise': [[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1],[0.1, 0.1]], 'anc_size': 1000, 'offset': 0}, \
]

#########################################################################################################
all_exp_settings = \
{
    'synth': synth, \
    'mimic': mimic, \
}

##########################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')
