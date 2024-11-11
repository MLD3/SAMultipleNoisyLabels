'''
hierarchical neural network
models labelers separately
combines predictions by averaging
'''

import numpy as np
import itertools
import torch
import torch.nn as nn

import util


###################################################################################################
'''
the overall network
'''
class hierarch_sep_net(nn.Module):
    def __init__(self, hyperparams, data_params):
        super(hierarch_sep_net, self).__init__()
        
        self.num_labs = data_params['num_labelers']

        self.num_steps = data_params['num_steps'] + 1 
        self.num_feats = data_params['num_feats']
        self.layer_div = data_params['layer_div'] 
        self.num_layers = hyperparams['n_layer']
        self.layer_size = np.floor(hyperparams['layer_s']).astype(int)
        
        self.activation = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)
        
        self.shared = nn.Linear(self.num_feats, self.num_feats)
        
        self.models = []
        self.model_oc = []
        self.all_layers = []
        for k in range(self.num_labs):
            hidden = []
            hidden.append(nn.Linear(self.num_feats, self.layer_size))
            for _ in range(self.num_layers - 1):
                hidden.append(nn.Linear(self.layer_size, self.layer_size))
            output = nn.Linear(self.layer_size, self.num_steps)
            
            hidden_oc = [nn.Linear(self.num_feats, self.layer_size)]
            output_oc = nn.Linear(self.layer_size, 2)
        
            mod_layers = hidden + [output]
            oc_layers = hidden_oc + [output_oc]
            
            self.all_layers += mod_layers + oc_layers
            self.models.append(mod_layers)
            self.model_oc.append(oc_layers)
           
        self.all_layers = nn.ModuleList(self.all_layers)
        
        self.init_weights()
    
    def init_weights(self):
        for i in range(len(self.all_layers)):
            nn.init.kaiming_uniform_(self.all_layers[i].weight)
   
    def mod_forward(self, inp, mod_num, return_extra=False, extra_args=None):
        out_occur = self.model_oc[mod_num][0](inp) #predict if it occurs outside horizon
        out_occur = self.activation(out_occur)
        out_occur = self.model_oc[mod_num][1](out_occur)
        pred_occur = self.softmax(out_occur)
        
        out = self.models[mod_num][0](inp)
        for i in range(1, len(self.models[mod_num])-1):
            out = self.activation(out)
            out = self.models[mod_num][i](out)
        out = self.models[mod_num][-1](out)
        preds = self.softmax(out)
        
        if return_extra:
            preds = self.softmax(out) * pred_occur[:, 0].reshape(-1, 1)
            preds = torch.cat([preds, pred_occur[:, 1].reshape(-1, 1)], dim=1)
            return {'input': inp, 'preds': preds, 'emb': out}
        
        preds = self.softmax(out) * pred_occur[:, 0].reshape(-1, 1)
        preds = torch.cat([preds, pred_occur[:, 1].reshape(-1, 1)], dim=1)
        return preds
    
    def forward(self, inp, return_extra=False, extra_args=None):
        final_preds = []
        embs = []
        final_final_preds = 0
        
        shared_rep = self.shared(inp) 
        
        for i in range(self.num_labs):
            mod_out = self.mod_forward(shared_rep, i, return_extra, extra_args)
            if return_extra:
                final_preds.append(mod_out['preds'])
                embs.append(mod_out['emb'])
                final_final_preds += mod_out['preds']
            else:
                final_preds.append(mod_out)
                final_final_preds += mod_out
        final_final_preds /= self.num_labs
        if return_extra:
            return {'input': inp, 'preds': final_final_preds, 'mod_preds': final_preds, 'emb': embs}
        
        return final_final_preds.detach().cpu().numpy() 
    
    def get_parameters(self):
        params = []
        for i in range(len(self.all_layers)):
            params.append(self.all_layers[i].parameters())
        
        params = itertools.chain.from_iterable(params)        
        return params
    

###################################################################################################
'''
pretty much cross entropy, making a separate class for this so that the proposed loss can be customized
'''
class hierarch_sep_loss(nn.Module):
    def __init__(self, hyperparams):
        super(hierarch_sep_loss, self).__init__()
        
        self.loss = nn.CrossEntropyLoss()
        self.cen_weight = hyperparams['cen_weight']

    def forward_mod(self, outputs, labs, mod_num, extra_args=None):
        preds = outputs['mod_preds'][mod_num]
        pre_pred = outputs['emb'][mod_num] #presoftmax output
        cen_in = extra_args['cen_in'][mod_num] 
            
        censored = np.where(cen_in == 1)[0]
        uncensored = np.where(cen_in == 0)[0]
        num_cen, num_uncen = censored.shape[0], uncensored.shape[0]
        num_ex = labs[0].shape[0]
   
        bin_indexes = util.to_gpu(labs[mod_num][uncensored].type(torch.LongTensor)).reshape(-1, 1)
        uncen_loss = 0
        if bin_indexes.shape[0] > 0:
            uncen_loss = -torch.sum(torch.log(torch.gather(preds[uncensored, :], 1, bin_indexes)+1e-10)) / max(1, num_uncen)
        
        cen_loss = util.to_gpu(torch.Tensor([0])) 
        cen_times = np.unique(labs[0][censored].cpu().numpy())
        cen_preds = preds[censored, :]
        for i in cen_times:
            last_obs = np.where(labs[0][censored] == i)[0].astype(int)
            cen_loss += torch.sum(-torch.log(torch.sum(cen_preds[last_obs, :][:, int(i+1):], dim=1)+1e-10)) / max(num_cen, 1) 
        
        loss = (num_uncen/num_ex)*uncen_loss + (num_cen/num_ex)*cen_loss*self.cen_weight
        
        return loss
        
    def forward(self, outputs, labs, extra_args=None):
        num_mods = len(outputs['mod_preds'])
        loss = 0
        for i in range(num_mods):
            mod_loss = self.forward_mod(outputs, labs, i, extra_args)
            loss += mod_loss
        
        return loss
        

###################################################################################################
'''
training function
'''
def get_model(dataset_name, dataset_package, approach, data_params, hyperparams):
    train_data, test_data, val_data = dataset_package[0], dataset_package[1], dataset_package[2]
    dataset = {'train': train_data, 'val': val_data}
    
    model = hierarch_sep_net(hyperparams, data_params)
    loss_fx = hierarch_sep_loss(hyperparams)
        
    model, val_loss, ep = util.train_model(model, loss_fx, hyperparams, dataset, data_params, approach)

    return model, val_loss, ep


###################################################################################################
'''
main block 
'''
if __name__ == '__main__':
    print(':)')
