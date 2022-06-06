from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import *
from utils_from_GCNII import *
import uuid
import scipy.io as sio
from GPCN  import *
from Data_split_object import Data_split
import scipy.io as sio
from data_utils_non_homoginous2 import *
# Training settings


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=4, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='twitch-eDE', help='dateset')
parser.add_argument('--dev', type=int, default=2, help='device id')
parser.add_argument('--init_layers', action='store_true', default=1)
parser.add_argument('--res_scale', type=float, default=0.5)
parser.add_argument('--T', type=int, default=1)
parser.add_argument('--w_relu',  type=bool, default=False)
parser.add_argument('--norm_first', type=bool, default=True)
parser.add_argument('--norm_residual', type=bool, default=True)
parser.add_argument('--mlpX', type=int, default=1)
parser.add_argument('--model_type', default='GPCN', help='GPCN,GPCN-LINK,AGPCN,AGPCN-LINK')
parser.add_argument('--last_activation', default='log_softmax', help='log_softmax ot -')
args = parser.parse_args()


no_samples = 10


res_array = np.zeros([len(lr), len(hidden) , len(wd) , len(res_scale) , len(T) ,   len(drop), len(drop_non_residual)   , no_samples])
test_array = np.zeros([len(lr), len(hidden) , len(wd) , len(res_scale) , len(T) ,   len(drop), len(drop_non_residual)   , no_samples])
cudaid = "cuda:" + str(args.dev)
device = torch.device(cudaid)
checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'
print(cudaid, checkpt_file)


split = torch.load('./data_splits_large2/' + str(args.data) + '.pt')

#print(split)
adj = split.adj
labels = split.y
#labels = list(labels[0])
if len(labels.shape) == 1:
    labels = labels.unsqueeze(1)
num_classes = split.num_classes 
features = split.x
#print(labels)

adj = adj.to(device)
labels = torch.LongTensor(labels).to(device)
features = torch.FloatTensor(features).to(device)


   
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
acc_list = []
cf_list = []

for j in range(no_samples):

    def train_step(model,optimizer,features,labels,adj,idx_train):
        model.train()
        optimizer.zero_grad()
        output = model(features)
        if labels.shape[1] == 1:
            labels = F.one_hot(labels, 2).squeeze(1).to(torch.float) # labels.max() + 1).squeeze(1)
        else:
            labels = labels
#                   print(output.size())
#                   print(labels)
#                   print(output)
#                   print(output.size())
        acc_train = eval_rocauc(labels[idx_train].to(torch.float).to(device), output[idx_train])
        loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].to(device)) 
        loss_train.backward()
        optimizer.step()
        return loss_train.item(),acc_train.item()


    def validate_step(model,features,labels,adj,idx_val):
        model.eval()
        with torch.no_grad():
            output = model(features)
            if labels.shape[1] == 1:
                labels = F.one_hot(labels, 2).squeeze(1).to(torch.float) # labels.max() + 1).squeeze(1)
            else:
                labels = labels                            
            output = model(features)
            loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].to(device))
            acc_val = eval_rocauc( labels[idx_val].to(device), output[idx_val])
            return loss_val.item(),acc_val.item()

    def test_step(model,features,labels,adj,idx_test):
        model.load_state_dict(torch.load(checkpt_file))
        model.eval()
        with torch.no_grad():
            output = model(features)
            if labels.shape[1] == 1:
                labels = F.one_hot(labels, 2).squeeze(1).to(torch.float) # labels.max() + 1).squeeze(1)
            else:
                labels = labels
            loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test].to(device))
            acc_test = eval_rocauc(labels[idx_test].to(device),output[idx_test])
            return loss_test.item(),acc_test.item()


    def train(j):

        
        #features, labels, adj = d.x, d.y,d.adj
        idx_train = split.idx_train[j]  #split[j]['train'] 
        idx_val = split.idx_val[j] #split[j]['valid']
        idx_test = split.idx_test[j]  #split[j]['test']
        
        num_labels = num_classes

        #  = GPCN_mlpX(nclass=num_labels, n_features=features.shape[1], nc_hidden=hidden_, num_nodes = features.shape[0],  T = T_,  A1 = adj, feat_layers=args.mlpX, res_scale=res_scale_, drop_non_residual = drop_, drop_residual = drop_, criterion = '-',  w_relu = True,  w_relu_A = False).to(device)

        if args.model_type == 'GPCN':
            model  = GPCN(nclass=num_labels, n_features=features.shape[1], nc_hidden= args.hidden, num_nodes = features.shape[0], T = args.T, A1 = adj, feat_layers= args.mlpX, res_scale=args.res_scale, drop_non_residual = args.dropout, drop_residual = args.dropout,  criterion = args.last_activation, w_relu = True,  w_relu_A = False).to(device)
        elif args.model_type == 'GPCN-LINK':
            model  = GPCN_LINK(nclass=num_labels, n_features=features.shape[1], nc_hidden= args.hidden, num_nodes = features.shape[0], T = args.T, A1 = adj, feat_layers= args.mlpX, res_scale=args.res_scale, drop_non_residual = args.dropout, drop_residual = args.dropout,  criterion = args.last_activation, w_relu = True,  w_relu_A = False).to(device)
        elif args.model_type == 'AGPCN':
            model  = AGPCN(nclass=num_labels, n_features=features.shape[1], nc_hidden= args.hidden, num_nodes = features.shape[0], T = args.T, A1 = adj, feat_layers= args.mlpX,  drop_non_residual = args.dropout, drop_residual = args.dropout,  criterion = args.last_activation, w_relu = True,  w_relu_A = False).to(device)
        elif args.model_type == 'AGPCN-LINK':
            model  = AGPCN_LINK(nclass=num_labels, n_features=features.shape[1], nc_hidden= args.hidden, num_nodes = features.shape[0], T = args.T, A1 = adj, feat_layers= args.mlpX, drop_non_residual = args.dropout, drop_residual = args.dropout,  criterion = args.last_activation , w_relu = True,  w_relu_A = False).to(device)
        else:
            print('Unspcified model type....')
            return


        optimizer = optim.Adam(model.parameters(), lr=lr_, weight_decay=wd_)

        bad_counter = 0
        best = 999999999
        for epoch in range(args.epochs):
            loss_tra,acc_tra = train_step(model,optimizer,features,labels,adj,idx_train)
            loss_val,acc_val = validate_step(model,features,labels,adj,idx_val)

            if(epoch+1)%50 == 0:
                print('Epoch:{:04d}'.format(epoch+1),
                    'train',
                    'loss:{:.3f}'.format(loss_tra),
                    'acc:{:.2f}'.format(acc
                    'loss:{:.3f}'.format(loss_val),_tra*100),
                    '| val',
                    'acc:{:.2f}'.format(acc_val*100))

            if loss_val < best:
                best = loss_val
                torch.save(model.state_dict(), checkpt_file)
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                break

        #acc = test_step(model,features,labels,adj,idx_test)[1]
        model.load_state_dict(torch.load(checkpt_file))
        loss_val, acc_val = validate_step(model, features, labels, adj, idx_val)
        loss_test, acc_test = test_step(model, features, labels, adj, idx_test)
        return acc_val*100, acc_test*100 


    
    accs, accs_test= train(j)
    acc_list.append(acc_test)


    #filename = './full_GPCN_newdata_gridsearch_new/'+str(args.data) + '_GPCN_BCE_'   + '_' + str(args.seed) + '_' + str(lr)  + '_' + str(hidden)  + '_' + str(res_scale)   + '_' + str(wd)   + '_' + str(T)  + '_' + str(drop)   + '_' + str(drop_non_residual)  + '_' + str(args.mlpX)  + '_.mat'
    #sio.savemat(filename, {'res': res_array, 'test': test_array})
  
print("Accuracy : " + str(np.mean(acc_list)) + ' (' + str(np.std(acc_list)) +')')

