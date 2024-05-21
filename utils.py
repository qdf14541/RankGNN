import os
import argparse
import yaml
import numpy as np
import random
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, recall_score
from sklearn.metrics import precision_score


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp')
    parser.add_argument('--config', type=str, default='yelpv4.yaml')
    args = parser.parse_args()
    config_path = './config/'+args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print_config(config)
    args = argparse.Namespace(**config)
    
    return args

class EarlyStop():
    def __init__(self, early_stop, if_more=True) -> None:
        self.best_eval = 0
        self.best_epoch = 0
        self.if_more = if_more
        self.early_stop = early_stop
        self.stop_steps = 0
    
    def step(self, current_eval, current_epoch):
        do_stop = False
        do_store = False
        if self.if_more:
            if current_eval > self.best_eval:
                self.best_eval = current_eval
                self.best_epoch = current_epoch
                self.stop_steps = 1
                do_store = True
            else:
                self.stop_steps += 1
                if self.stop_steps >= self.early_stop:
                    do_stop = True
        else:
            if current_eval < self.best_eval:
                self.best_eval = current_eval
                self.best_epoch = current_epoch
                self.stop_steps = 1
                do_store = True
            else:
                self.stop_steps += 1
                if self.stop_steps >= self.early_stop:
                    do_stop = True
        return do_store, do_stop

def conf_gmean(conf):
	tn, fp, fn, tp = conf.ravel()
	return (tp*tn/((tp+fn)*(tn+fp)))**0.5
def prob2pred(prob, threshhold=0.5):
    pred = np.zeros_like(prob, dtype=np.int32)
    pred[prob >= threshhold] = 1
    pred[prob < threshhold] = 0
    return pred
def evaluate(labels, logits, result_path = ''):
    probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()
    preds = logits.argmax(1).cpu().numpy()
    if len(result_path)>0:
        np.save(result_path+'_result_preds', preds)
        np.save(result_path+'_result_probs', probs)
    conf = confusion_matrix(labels, preds)
    recall = recall_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    auc = roc_auc_score(labels, probs)
    gmean = conf_gmean(conf)
    precision = precision_score(labels, preds)
    return f1_macro, auc, gmean, recall, precision

def hinge_loss(labels, scores):
    margin = 1
    ls = labels*scores
    
    loss = F.relu(margin-ls)
    loss = loss.mean()
    return loss

def rank_loss(labels, scores, ratio=0.3, mask=None):

    preds = scores
    y = labels.reshape((-1, 1))
    p = preds.reshape((-1, 1))
    length = p.shape[0]
    
    prob = torch.softmax(p, dim=0)
    prob = prob.squeeze(-1)
    prob = prob + 1e-5

    indexs = torch.multinomial(prob, int(length*ratio), replacement=False)
    # print('indexs shape', indexs.shape)
    # print(indexs)
    y = y[indexs]
    p = p[indexs]
    mask = mask[indexs, :][:, indexs]
    mask_inter = torch.sum(mask, dim=1).bool()
    # print(y.shape)
    Inter_label = torch.clone(y).detach().reshape(-1, 1)
    Inter_pre = torch.clone(p).detach().reshape(-1, 1)
    mask_inter = mask_inter.reshape(-1,1)

    y_diff = y - y.t()
    y_diff = y_diff * mask
    p_diff = p - p.t()
    
    m = (y_diff != 0)
    if len(m) == 0:
        print('No heterophilic connections in the sampling.')
        loss = torch.zeros_like(y).reshape(-1, 1)[0]
        return loss    

    p_diff = p_diff[m]
    y_diff = y_diff[m]
    tij = (1.0 + torch.sign(y_diff)) / 2.0
    loss = torch.nn.BCEWithLogitsLoss()(p_diff.float(), tij.float())
    return loss, Inter_label, Inter_pre, mask_inter

def normalize(mx):

	"""
		Row-normalize sparse matrix
		Code from https://github.com/williamleif/graphsage-simple/
	"""
	rowsum = np.array(mx.sum(1)) + 0.01
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx

