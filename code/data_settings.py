'''
record of all settings for datasets
'''

synth = \
{
    'name': 'synthetic', \
    'num_data': 5000, \
    'num_feats': 100, \
    'num_steps': 100, \
    'min_ep': 20, \
    'max_ep': 200, \
    #'layer_div': [1, 1, 10, 5, 2], \
}

mimic = \
{
    'name': 'mimic', \
    'min_ep': 20, \
    'max_ep': 200, \
    #'layer_div': [1, 1, 10, 5, 2], \
}

##########################################################################################################
all_settings = \
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
