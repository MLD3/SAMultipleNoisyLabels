'''
proposed
'''

import numpy as np
import itertools
import torch
import torch.nn as nn

import util

import copy


###################################################################################################
'''
the overall network
'''
class proposed3_net(nn.Module):
    def __init__(self, hyperparams, data_params, obs_only, pred_err, marg_err_preds):
        super(proposed3_net, self).__init__()

        self.num_steps = data_params['num_steps'] 
        self.num_feats = data_params['num_feats']
        self.num_layers = hyperparams['n_layer']
        self.layer_size = np.floor(hyperparams['layer_s']).astype(int)
        self.train_final = False
        self.obs_only = obs_only
        self.marginalize_preds = marg_err_preds
        self.pred_err = pred_err
        
        self.activation = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)
        
        self.hidden = []
        self.hidden.append(nn.Linear(self.num_feats, self.layer_size))
        for _ in range(self.num_layers - 1):
            self.hidden.append(nn.Linear(self.layer_size, self.layer_size))
        self.output = nn.Linear(self.layer_size, self.num_steps)
        
        self.output_gt1 = nn.Linear(self.num_steps+self.num_feats, self.num_steps)
        self.output_gt = nn.Linear(self.num_steps, self.num_steps)
        
        self.hidden_oc = [nn.Linear(self.num_feats, self.layer_size)]
        self.output_oc = nn.Linear(self.layer_size, 2)
        
        self.layers = self.hidden + [self.output, self.output_gt] + self.hidden_oc + [self.output_oc] \
                    + [self.output_gt1]
        
        self.error_marg = self.num_steps + 1 
        if self.pred_err:
            self.hidden_error = nn.Linear(self.num_feats, self.layer_size)
            self.output_error = nn.Linear(self.layer_size, self.error_marg*2 - 1)
            self.layers += [self.hidden_error, self.output_error]
            
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
        
        out_err = 0
        if self.pred_err:
            out_err = self.hidden_error(inp)
            out_err = self.activation(out_err)
            out_err = self.softmax(self.output_error(out_err))
        
        if self.train_final:
            out = out.detach().cpu().numpy()
            pred_occur = pred_occur.detach().cpu().numpy()
            out = util.to_gpu(torch.Tensor(out))  
            pred_occur = util.to_gpu(torch.Tensor(pred_occur))
        gt_preds = self.output_gt1(torch.cat([self.activation(out), inp], dim=1))
        gt_preds = self.output_gt(self.activation(gt_preds))
        
        if return_extra:
            preds = self.softmax(out) * pred_occur[:, 0].reshape(-1, 1)
            preds = torch.cat([preds, pred_occur[:, 1].reshape(-1, 1)], dim=1)
            gt_preds = self.softmax(gt_preds) * pred_occur[:, 0].reshape(-1, 1)
            gt_preds = torch.cat([gt_preds, pred_occur[:, 1].reshape(-1, 1)], dim=1)
            return {'input': inp, 'preds': preds, 'gt_preds': gt_preds, 'error_preds': out_err}
        
        preds = self.softmax(out) * pred_occur[:, 0].reshape(-1, 1)
        preds = torch.cat([preds, pred_occur[:, 1].reshape(-1, 1)], dim=1)
        gt_preds = self.softmax(gt_preds) * pred_occur[:, 0].reshape(-1, 1)
        gt_preds = torch.cat([gt_preds, pred_occur[:, 1].reshape(-1, 1)], dim=1)
        
        if (self.obs_only or not self.train_final) and not self.marginalize_preds:
            return preds.detach().cpu().numpy()
        elif self.marginalize_preds:
            preds = preds.detach().cpu().numpy()
            out_err = out_err.detach().cpu().numpy()
            marg_preds = np.zeros(preds.shape)
            num_times = preds.shape[1]
            err_vals = np.arange(-num_times+1, num_times)
            for i in range(num_times):
                #max(i-2, 0)/2, max(i-1, 0)/1, i/0, min(T, i+1)/-1, min(T, i+2)/-2, ...
                #P(e=1) = 0.25, P(e=2) = 0.25, P(e=3) = 0.5
                #P(g=-2) = 0.1, P(g=-1) = 0.2, P(g=0) = 0.3, P(g=1) = 0.1, P(g=2) = 0.3
                #P(e_gt=1) = P(e=1)P(g=0) + P(e=2)P(g=-1) + P(e=3)P(g=-2)
                #P(e_gt=2) = P(e=1)P(g=1) + P(e=2)P(g=0) + P(e=3)P(g=-1)
                #P(e_gt=3) = P(e=1)P(g=2) + P(e=2)P(g=1) + P(e=3)P(g=0)
                #sum of the P(e_gt's) doesn't give 1 because not all of the e/g pairs are valid
                #missing: [P(e=1)P(g=-1), P(e=1)P(g=-2), P(e=2)P(g=-2)] -> lump to 1 
                #[P(e=2)P(g=2), P(e=3)P(g=1), P(e=3)P(g=2)] -> lump to 3
                #marg_preds_i = np.zeros((preds.shape[0],))
                for j in range(len(err_vals)):
                    err_j = err_vals[j]
                    time_j = min(max(i + err_j, 0), num_times-1)
                    marg_preds[:, time_j] += preds[:, i] * out_err[:, j]
            return marg_preds
        
        return gt_preds.detach().cpu().numpy()
    
    def get_parameters(self):
        params = []
        for i in range(len(self.layers)):
            params.append(self.layers[i].parameters())
        
        params = itertools.chain.from_iterable(params)        
        return params
    

###################################################################################################
'''
pretty much cross entropy, making a separate class for this so that the proposed loss can be customized
'''
class proposed3_loss(nn.Module):
    def __init__(self, hyperparams, use_ce, exp_only, no_neg, have_gt, marg, horizon):
        super(proposed3_loss, self).__init__()
        
        self.loss = nn.CrossEntropyLoss()
        self.cen_weight = hyperparams['cen_weight']
        self.exp_weight = hyperparams['exp_weight']
        self.ent_weight = hyperparams['ent_weight']
        self.train_final = False
        self.horizon = horizon
        
        self.have_gt = have_gt
        self.use_ce = use_ce
        self.exp_only = exp_only
        self.no_neg = no_neg
        self.marg = marg

    def forward_lab(self, outputs, labs, lab_num, extra_args=None):
        preds = outputs['preds']
        cen_in = extra_args['cen_in'][lab_num] 
        num_steps = torch.max(labs[0])
            
        censored = np.where(cen_in == 1)[0]
        uncensored = np.where(cen_in == 0)[0]
        
        num_cen, num_uncen = censored.shape[0], uncensored.shape[0]
        num_ex = num_cen + num_uncen
        
        #uncensored for that labeler
        uncen_loss = util.to_gpu(torch.Tensor([0]))
        if num_uncen > 0:
            bin_indexes = util.to_gpu(labs[lab_num][uncensored].type(torch.LongTensor)).reshape(-1, 1)
            uncen_loss = -torch.sum(torch.log(torch.gather(preds[uncensored, :]+1e-10, 1, bin_indexes))) / max(num_uncen, 1)
        
        #censored for that labelers
        cen_loss = util.to_gpu(torch.Tensor([0])) 
        if num_cen > 0:
            cen_times = np.unique(labs[lab_num][censored].cpu().numpy()) 
            cen_preds = preds[censored, :]
            for i in cen_times:
                last_obs = np.where(labs[lab_num][censored] == i)[0].astype(int) 
                cen_loss += torch.sum(-torch.log(torch.sum(cen_preds[last_obs, :][:, int(i+1):]+1e-10, dim=1))) / max(num_cen, 1) 
          
        #put everything together 
        loss = (num_uncen/num_ex)*uncen_loss + (num_cen/num_ex)*cen_loss*self.cen_weight
        return loss
        
    def forward_err(self, outputs, labs, lab_num, extra_args):
        preds = outputs['preds']
        error_preds = outputs['error_preds']
        cen_in = extra_args['cen_in'][lab_num] 
        
        anc = extra_args['anc_in']
        gt_times = extra_args['gt_times']
        gt_cen = extra_args['gt_cen']
         
        anchor = np.where(anc == 1)[0]
        censored_gt = np.intersect1d(np.where(gt_cen == 1)[0], anchor)
        uncensored_gt = np.intersect1d(np.where(gt_cen == 0)[0], anchor)
        
        uncensored_obs = np.where(cen_in == 0)[0]
        censored_obs = np.where(cen_in == 1)[0]
        fp_gt = np.intersect1d(uncensored_obs, censored_gt)
        tn_gt = np.intersect1d(censored_obs, censored_gt)
        fn_gt = np.intersect1d(censored_obs, uncensored_gt)
        tp_gt = np.intersect1d(uncensored_obs, uncensored_gt)
        
        num_neg = censored_gt.shape[0]
        
        censored = censored_gt
        
        #event occurrence
        neg_loss = util.to_gpu(torch.Tensor([0]))
        not_anc = np.array([])
        
        if num_neg > 0:
            neg_loss = -torch.sum(torch.log(preds[censored_gt, -1]+1e-10))*self.cen_weight 
            neg_loss += -torch.sum(torch.log(1-preds[uncensored_gt, -1]+1e-10)) 
        
        gpu_lab = util.to_gpu(labs[lab_num].type(torch.LongTensor))
        #true positives 
        tp_loss = util.to_gpu(torch.Tensor([0]))
        if tp_gt.shape[0] > 0:
            obs_indexes = (gpu_lab[tp_gt]).reshape(-1, 1)
            gt_indexes = util.to_gpu(gt_times[tp_gt].type(torch.LongTensor)).reshape(-1, 1)
            err_indexes = gt_indexes - obs_indexes + (self.horizon)
            tp_loss = -torch.sum(torch.log(torch.gather(error_preds[tp_gt, :], 1, err_indexes)+1e-10)) 
            
        #false negatives
        fn_loss = util.to_gpu(torch.Tensor([0]))
        fn_gt = np.setdiff1d(fn_gt, np.where(labs[lab_num] < self.horizon)[0])
        if fn_gt.shape[0] > 0:
            obs_indexes = (gpu_lab[fn_gt]).reshape(-1, 1)   
            gt_indexes = util.to_gpu(gt_times[fn_gt].type(torch.LongTensor)).reshape(-1, 1)
            err_indexes = gt_indexes 
            fn_loss = -torch.sum(torch.log(torch.gather(error_preds[fn_gt, :], 1, err_indexes)+1e-10))
        
        #true negatives
        tn_loss = util.to_gpu(torch.Tensor([0]))
        tn_gt = np.setdiff1d(tn_gt, np.where(labs[lab_num] < self.horizon)[0])
        if tn_gt.shape[0] > 0:
            tn_loss = -torch.sum(torch.log(error_preds[tn_gt, self.horizon]+1e-10)) 
        
        #false positives
        fp_loss = util.to_gpu(torch.Tensor([0]))
        if fp_gt.shape[0] > 0:
            obs_indexes = (gpu_lab[fp_gt]).reshape(-1, 1)
            err_indexes = (self.horizon) - obs_indexes + (self.horizon)
            fp_loss = -torch.sum(torch.log(torch.gather(error_preds[fp_gt, :], 1, err_indexes)+1e-10)) 
        
        loss = (tp_loss + fn_loss + (tn_loss + fp_loss)*self.cen_weight)/max(anchor.shape[0], 1) \
               + neg_loss/(max(anchor.shape[0], 1))
        if self.no_neg:
                loss = (tp_loss + fn_loss + tn_loss + fp_loss)/max(anchor.shape[0], 1)
        return loss
    
    def forward_gt(self, outputs, labs, extra_args):
        preds = outputs['preds'].detach().cpu().numpy()
        preds = util.to_gpu(torch.Tensor(preds)) 
        gt_preds = outputs['gt_preds']
        num_sample = preds.shape[0]
        
        #get conditional probs
        denom = 1-preds[:, -1].reshape(-1, 1)
        denom_gt = 1-gt_preds[:, -1].reshape(-1, 1)
        denom[denom == 0] = 1
        denom_gt[denom_gt==0] = 1
        preds_cond = (preds[:, :-1] / denom) + 1e-10
        gt_preds_cond = (gt_preds[:, :-1] / denom_gt) + 1e-10
        if self.have_gt or self.no_neg:
            preds_cond = preds + 1e-10
            gt_preds_cond = gt_preds + 1e-10
        
        #expected value of obs distr preds - regular version and noisy version
        nums = util.to_gpu(torch.Tensor(np.arange(preds_cond.shape[1]).reshape(1, -1))) 
        exp_preds = torch.sum(preds_cond * nums, dim=1)
        
        #make expected values the same
        exp_preds_gt = torch.sum(gt_preds_cond * nums, dim=1)
        if self.have_gt or self.no_neg or self.marg:
            error_preds = util.to_gpu(torch.Tensor(outputs['error_preds'].detach().cpu().numpy()))
            err_nums = util.to_gpu(torch.Tensor(np.arange(-(self.horizon), self.horizon+1).reshape(1, -1)))
            exp_err = torch.sum(error_preds * err_nums, axis=1)
            exp_preds = exp_preds + exp_err
        exp_preds[exp_preds < 0] = 0
        exp_preds[exp_preds >= gt_preds_cond.shape[1]] = gt_preds_cond.shape[1] - 1 
        loss_exp = torch.sum(torch.square(exp_preds.detach() - exp_preds_gt)) / num_sample
        
        exp_preds2 = exp_preds 
        exp_preds2[exp_preds2 < 0] = 0
        exp_preds2[exp_preds2 >= gt_preds_cond.shape[1]] = gt_preds_cond.shape[1]-1
               
        exp_preds2 = util.to_gpu(exp_preds2.type(torch.LongTensor)).reshape(-1, 1)
        
        loss_ce = -torch.sum(torch.log(torch.gather(gt_preds_cond, 1, exp_preds2.detach()))) / gt_preds_cond.shape[0]
         
        if self.exp_only:
            return loss_exp
        elif not self.use_ce:
            return self.exp_weight*loss_exp + loss_ce
        
        #cross entropy over expected predictions
        if self.use_ce:
            return loss_ce 
        
    def forward(self, outputs, labs, extra_args=None):
        if self.train_final:
            loss = self.forward_gt(outputs, labs, extra_args)
            return loss
            
        num_labs = len(labs)
        loss = 0
        for i in range(num_labs):
            mod_loss = self.forward_lab(outputs, labs, i, extra_args)
            loss += mod_loss
            if self.have_gt or self.no_neg or self.marg:
                loss += self.forward_err(outputs, labs, i, extra_args)
        return loss 


###################################################################################################
'''
training function
'''
def get_model(dataset_name, dataset_package, approach, data_params, hyperparams):
    train_data, test_data, val_data = dataset_package[0], dataset_package[1], dataset_package[2]
    
    #make indicator vectors for time to events for everyone
    num_ex = train_data['feats'].shape[0]
    num_steps = data_params['num_steps']
    num_labs = data_params['num_labelers']
    horizon = data_params['horizon']
    
    #if only using a specific labeler
    obs_times, cen_in = [], []
    if approach[-4:] in ['err0', 'err1', 'err2', 'err3', 'err4', 'err5', 'err6', 'err7', 'err8', 'err9']:
        use_lab = int(approach[-1])
        for i in range(len(dataset_package)):
            obs_times.append(copy.deepcopy(dataset_package[i]['obs_times']))
            cen_in.append(copy.deepcopy(dataset_package[i]['cen_in']))
            new_times = dataset_package[i]['obs_times'][use_lab:use_lab+1]
            new_cens = dataset_package[i]['cen_in'][use_lab:use_lab+1]
            dataset_package[i]['obs_times'] = new_times
            dataset_package[i]['cen_in'] = new_cens
        num_labs = 1
   
    dataset = {'train': train_data, 'val': val_data}
    
    use_ce = 'ce' in approach
    exp_only = 'exp' in approach
    no_neg = 'noneg' in approach
    obs_only = 'obs' in approach
    pred_err = 'err' in approach
    marg_err = 'marg' in approach
    
    model = proposed3_net(hyperparams, data_params, obs_only, pred_err or no_neg, marg_err)
    loss_fx = proposed3_loss(hyperparams, use_ce, exp_only, no_neg, pred_err, marg_err, model.error_marg-1)
    
    model, val_loss, ep = util.train_model(model, loss_fx, hyperparams, dataset, data_params, approach)
    if obs_only or marg_err:
        return model, val_loss, ep
        
    model.train_final = True
    loss_fx.train_final = True
    model, val_loss, ep = util.train_model(model, loss_fx, hyperparams, dataset, data_params, approach) 
    
    if len(obs_times) == len(dataset_package):
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
