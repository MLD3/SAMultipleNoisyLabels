'''
feed forward neural network
'''

import copy
import numpy as np
import itertools
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import util


###################################################################################################
'''
the overall network
'''
class baseline1_net(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(baseline1_net, self).__init__()

        self.num_steps = data_params['num_steps'] 
        self.num_feats = data_params['num_feats']
        self.num_layers = hyperparams['n_layer']
        self.layer_size = np.floor(hyperparams['layer_s']).astype(int)
        
        self.activation = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)
        
        self.hidden = []
        self.hidden.append(nn.Linear(self.num_feats, self.layer_size))
        for _ in range(self.num_layers - 1):
            self.hidden.append(nn.Linear(self.layer_size, self.layer_size))
        self.output = nn.Linear(self.layer_size, self.num_steps)
        
        self.hidden_oc = [nn.Linear(self.num_feats, self.layer_size)]
        self.output_oc = nn.Linear(self.layer_size, 2)
        
        self.layers = self.hidden + [self.output] + self.hidden_oc + [self.output_oc]
        self.layers = nn.ModuleList(self.layers)
           
        self.init_weights()
    
    def init_weights(self):
        for i in range(len(self.layers)):
            nn.init.kaiming_uniform_(self.layers[i].weight)
    
    def forward(self, inp, return_extra=False, extra_args=None):
        out_occur = self.hidden_oc[0](inp) #predict if it occurs outside horizon
        out_occur = self.activation(out_occur)
        out_occur = self.output_oc(out_occur)
        pred_occur = self.softmax(out_occur)
        
        out = self.hidden[0](inp)
        for i in range(1, len(self.hidden)):
            out = self.activation(out)
            out = self.hidden[i](out)
        out = self.output(out)
        
        if return_extra:
            preds = self.softmax(out) * pred_occur[:, 0].reshape(-1, 1)
            preds = torch.cat([preds, pred_occur[:, 1].reshape(-1, 1)], dim=1)    
            return {'input': inp, 'preds': preds, 'emb': out}
        
        preds = self.softmax(out) * pred_occur[:, 0].reshape(-1, 1)
        preds = torch.cat([preds, pred_occur[:, 1].reshape(-1, 1)], dim=1)
        return preds.detach().cpu().numpy() 
    
    def get_parameters(self):
        params = []
        for i in range(len(self.layers)):
            params.append(self.layers[i].parameters())
        
        params = itertools.chain.from_iterable(params)        
        return params
        
    def plot_surv_curves(self, output, ttes):
        curves = util.get_surv_curve(output)
        print(curves.shape, output.shape)
        
        i = 0
        
        plt.plot(np.arange(curves.shape[1]), curves[i, :], marker='o', label='Direct')
        plt.xlabel('Time')
        plt.ylabel('Probability of Survival')
        plt.title('Time to Event = ' + str(int(ttes[0][i])))
        plt.savefig('plots/feed_for_curve.png')
    

###################################################################################################
'''
pretty much cross entropy, making a separate class for this so that the proposed loss can be customized
'''
class baseline1_loss(nn.Module):
    def __init__(self, hyperparams, augment=False, ground_truth=False):
        super(baseline1_loss, self).__init__()
        
        self.loss = nn.CrossEntropyLoss()
        self.augment = augment
        self.gt = ground_truth
        
        self.cen_weight = hyperparams['cen_weight']

    def forward(self, outputs, labs, extra_args=None):
        preds = outputs['preds']
        pre_pred = outputs['emb'] #presoftmax output
        loss = util.to_gpu(torch.Tensor([0]))
        cen_in = extra_args['cen_in'][0] 
        
        censored = np.where(cen_in == 1)[0]
        uncensored = np.where(cen_in == 0)[0]
        num_cen, num_uncen = censored.shape[0], uncensored.shape[0]
        num_ex = labs[0].shape[0]

        uncen_loss = util.to_gpu(torch.Tensor([0])) 
        if uncensored.shape[0] > 0:
            bin_indexes = util.to_gpu(labs[0][uncensored].type(torch.LongTensor)).reshape(-1, 1)
            uncen_loss = -torch.sum(torch.log(torch.gather(preds[uncensored, :], 1, bin_indexes)+1e-10)) / max(1, num_uncen)
         
        cen_loss = util.to_gpu(torch.Tensor([0])) 
        cen_times = np.unique(labs[0][censored].cpu().numpy())
        cen_preds = preds[censored, :]
        for i in cen_times:
            last_obs = np.where(labs[0][censored] == i)[0].astype(int) 
            cen_loss += torch.sum(-torch.log(torch.sum(cen_preds[last_obs, :][:, int(i+1):], dim=1)+1e-10)) / max(1, num_cen) 
        loss = (num_uncen/num_ex)*uncen_loss + (num_cen/num_ex)*cen_loss*self.cen_weight
        
        return loss


###################################################################################################
'''
training function
'''
def get_model(dataset_name, dataset_package, approach, data_params, hyperparams):
    approach_subtype = approach.split('_')[-1] 
    obs_times, cen_in = [], []
    
    for i in range(len(dataset_package)):
        obs_times.append(copy.deepcopy(dataset_package[i]['obs_times']))
        cen_in.append(copy.deepcopy(dataset_package[i]['cen_in']))
        if approach_subtype == 'gt':
            new_times = dataset_package[i]['gt_obs_times']
            new_cens = dataset_package[i]['gt_cen_in']
        elif approach_subtype == 'min':
            new_times, new_cens = util.aggregate(dataset_package[i]['obs_times'], dataset_package[i]['cen_in'], 'min')
        elif approach_subtype == 'max':
            new_times, new_cens = util.aggregate(dataset_package[i]['obs_times'], dataset_package[i]['cen_in'], 'max')
        elif approach_subtype == 'avgnaive':
            new_times, new_cens = util.aggregate(dataset_package[i]['obs_times'], dataset_package[i]['cen_in'], 'avg')
        elif approach_subtype == 'avgvote':
            new_times, new_cens = util.aggregate(dataset_package[i]['obs_times'], dataset_package[i]['cen_in'], 'avg', naive=False)
        elif approach_subtype == 'avgcen':
            new_times, new_cens = util.aggregate(dataset_package[i]['obs_times'], dataset_package[i]['cen_in'], 'avg', naive='Part')
        elif approach_subtype == 'rand':
            new_times, new_cens = util.aggregate(dataset_package[i]['obs_times'], dataset_package[i]['cen_in'], 'rand')
        else: #using one of the labelers by itself as ground truth
            use_lab = int(approach_subtype)
            new_times = dataset_package[i]['obs_times'][use_lab:use_lab+1]
            new_cens = dataset_package[i]['cen_in'][use_lab:use_lab+1]
        
        dataset_package[i]['obs_times'] = new_times
        dataset_package[i]['cen_in'] = new_cens
    
    #bias of aggregated label wrt ground truth in train data
    tte_diff = []
    num_train = dataset_package[0]['gt_times'].shape[0]
    for i in range(num_train):
        if dataset_package[0]['cen_in'][0][i] == 1:
            continue
        agg_tte = dataset_package[0]['obs_times'][0][i]
        gt_tte = dataset_package[0]['gt_times'][i]
        tte_diff.append(agg_tte - gt_tte)
    print('num censored when aggregated', num_train - len(tte_diff), num_train)
    print('num uncensored when aggregated', len(tte_diff), num_train)
    if len(tte_diff) > 0:
        print('aggregation bias [0, 25, 50, 75, 100] percentiles', np.percentile(tte_diff, [0, 25, 50, 75, 100]))   
        
    train_data, test_data, val_data = dataset_package[0], dataset_package[1], dataset_package[2]
    dataset = {'train': train_data, 'val': val_data}
    
    model = baseline1_net(hyperparams, data_params)
    use_gt = approach_subtype == 'gt'
    loss_fx = baseline1_loss(hyperparams, ground_truth=use_gt)
        
    model, val_loss, ep = util.train_model(model, loss_fx, hyperparams, dataset, data_params, approach)
    
    for i in range(len(dataset_package)):
        dataset_package[i]['obs_times'] = obs_times[i]
        dataset_package[i]['cen_in'] = cen_in[i]

    return model, val_loss, ep


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')
