import os
import numpy as np
import torch
import dgl
import torch.optim as optim
from model import *
from utils import *
from data_preprocess import getdata

import warnings
warnings.filterwarnings('ignore')

def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")

file_path = os.getcwd()

def train(args, seed):
    print('final seed:', seed)
    setup_seed(seed)
    prefix = 'seed'+str(seed)+'_'+str(args.version)+'_'
    print(args.data_path)
    print(args.dataset)
    getdata(os.path.join(file_path, args.data_path), args.dataset, prefix)
    device = torch.device(args.cuda)
    args.device = device
    dataset_path = args.data_path+prefix+args.dataset+'.dgl'
    model_path = args.result_path+prefix+args.dataset+'_model.pt'
    results = {'F1-macro':[],'AUC':[],'G-Mean':[],'recall':[], 'precision':[]}
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    print(args.result_path, os.path.exists(args.result_path))
    '''
    # load dataset and normalize feature
    '''
    dataset = dgl.load_graphs(dataset_path)[0][0]
    features = dataset.ndata['feature'].numpy()
    features = normalize(features)
    dataset.ndata['feature'] = torch.from_numpy(features).float()
    dataset = dataset.to(device)
    
    '''
    # train model
    '''
    print('Start training model...')
    model = Model(args, dataset)
    model = model.to(device)
    print(model)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    early_stop = EarlyStop(args.early_stop)
    Re_pre = []
    Re_y = []
    Con = []
    for e in range(args.epoch):
        
        model.train()
        loss, Inter_y, Inter_pre, connection = model.loss(dataset)
        Re_pre.append(Inter_pre.detach().cpu())
        Re_y.append(Inter_y.detach().cpu())
        Con.append(connection.detach().cpu())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.use_quant:
            model.set_num_updates(e)
        
        with torch.no_grad():
            '''
            # valid
            '''
            model.eval()
            valid_mask = dataset.ndata['valid_mask'].bool()
            valid_labels = dataset.ndata['label'][valid_mask].cpu().numpy()
            valid_logits = model(dataset)[valid_mask]
            valid_preds = valid_logits.argmax(1).cpu().numpy()
            f1_macro, auc, gmean, recall, precision = evaluate(valid_labels, valid_logits)
            
            if args.log:
                print('{}: Best Epoch: {}, Best valid AUC: {:.4f}, Loss: {:.4f}, Current valid: Recall: {:.4f}, F1_macro: {:.4f}, G-Mean: {:.4f}, AUC: {:.4f}, Precision: {:.4f}'.format(
    e, early_stop.best_epoch, early_stop.best_eval, loss.item(), recall, f1_macro, gmean, auc, precision
))
            do_store, do_stop = early_stop.step(auc, e)
            if do_store:
                torch.save(model, model_path)
            if do_stop:
                break
    print('End training')
    
    # save the intermediate results for checking
    torch.save(Re_y, args.result_path + 'sampling_' +str(seed)+'_label.pt')
    torch.save(Re_pre, args.result_path + 'sampling_' +str(seed)+'_pre.pt')
    torch.save(Con, args.result_path + 'sampling_' + str(seed) + '_connected.pt')
    '''
    # test model
    '''
    print('Test model...')
    model = torch.load(model_path)      
    with torch.no_grad():
        model.eval()
        test_mask = dataset.ndata['test_mask'].bool()
        test_labels = dataset.ndata['label'][test_mask]
        test_labels = test_labels.cpu().numpy()
        logits = model(dataset)[test_mask]
        logits = logits.cpu()
        test_result_path = args.result_path+args.dataset
        f1_macro, auc, gmean, recall, precision = evaluate(test_labels, logits, test_result_path)
        results['F1-macro'].append(f1_macro)
        results['AUC'].append(auc)
        results['G-Mean'].append(gmean)
        results['recall'].append(recall)
        results['precision'].append(precision)

        print('Test: F1-macro: {:.4f}, AUC: {:.4f}, G-Mean: {:.4f}, Recall: {:.4f}, Precision: {:.4f}'.format(
    f1_macro, auc, gmean, recall, precision
))
        return f1_macro, auc, gmean, recall, precision
    

if __name__ == '__main__':
    args = parse_args()
    
    f1_list = []
    auc_list = []
    gmean_list = []
    recall_list = []
    precision_list = []

    for seed in [72, 42, 448, 854, 29493, 88867, 20, 717, 123, 0]:
        print('running seed', seed)
        args.seed = seed
        f1_macro, auc, gmean, recall, precision = train(args, seed)
        
        f1_list.append(f1_macro)
        auc_list.append(auc)
        gmean_list.append(gmean)
        recall_list.append(recall)
        precision_list.append(precision)
        
    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list, ddof=1)
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list, ddof=1)
    gmean_mean, gmean_std = np.mean(gmean_list), np.std(gmean_list, ddof=1)
    recall_mean, recall_std = np.mean(recall_list), np.std(recall_list, ddof=1)
    precision_mean, precision_std = np.mean(precision_list), np.std(precision_list, ddof=1)    

    print("F1 list:", f1_list)
    print('AUC list:', auc_list)
    print("GMean list:", gmean_list)
    print("Recall list:", recall_list)
    print("Precision list:", precision_list)
    
    print("F1-Macro: {}+{}".format(f1_mean, f1_std))
    print("AUC: {}+{}".format(auc_mean, auc_std))
    print("G-Mean: {}+{}".format(gmean_mean, gmean_std))
    print("Recall: {}+{}".format(recall_mean, recall_std))
    print("Precision: {}+{}".format(precision_mean, precision_std))
    

