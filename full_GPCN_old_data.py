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
from utils import *
import uuid
import scipy.io as sio
from models  import *


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
parser.add_argument('--data', default='chameleon', help='dateset')
parser.add_argument('--dev', type=int, default=2, help='device id')
#parser.add_argument('--init_layers', action='store_true', default=1)
parser.add_argument('--T', type=int, default=1)
parser.add_argument('--w_relu',  type=bool, default=False)
parser.add_argument('--res_scale', type=float, default=1.0)
parser.add_argument('--mlpX', type=int, default=1)
parser.add_argument('--model_type', default='GPCN', help='GPCN,GPCN-LINK,AGPCN,AGPCN-LINK')
args = parser.parse_args()


no_samples = 10



cudaid = "cpu" #"cuda:" + str(args.dev)
device = torch.device(cudaid)
checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'
print(cudaid, checkpt_file)

   
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
        acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
        loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
        loss_train.backward()
        optimizer.step()
        return loss_train.item(),acc_train.item()


    def validate_step(model,features,labels,adj,idx_val):
        model.eval()
        with torch.no_grad():
            output = model(features)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
            acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
            return loss_val.item(),acc_val.item()

    def test_step(model,features,labels,adj,idx_test):
        model.load_state_dict(torch.load(checkpt_file))
        model.eval()
        with torch.no_grad():
            output = model(features)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
            acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
            return loss_test.item(),acc_test.item()


    def train(datastr,splitstr):
        adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels, adj_ = full_load_data(datastr,splitstr)
        features = features.to(device)
        adj = adj.to(device)

        if args.model_type == 'GPCN':
            model  = GPCN(nclass=num_labels, n_features=features.shape[1], nc_hidden= args.hidden, num_nodes = features.shape[0], T = args.T, A1 = adj, feat_layers= args.mlpX, res_scale=args.res_scale, drop_non_residual = args.dropout, drop_residual = args.dropout,  criterion = 'log_softmax', w_relu = True,  w_relu_A = False).to(device)
        elif args.model_type == 'GPCN-LINK':
            model  = GPCN(nclass=num_labels, n_features=features.shape[1], nc_hidden= args.hidden, num_nodes = features.shape[0], T = args.T, A1 = adj, feat_layers= args.mlpX, res_scale=args.res_scale, drop_non_residual = args.dropout, drop_residual = args.dropout,  criterion = 'log_softmax', w_relu = True,  w_relu_A = False).to(device)
        elif args.model_type == 'AGPCN':
            model  = AGPCN(nclass=num_labels, n_features=features.shape[1], nc_hidden= args.hidden, num_nodes = features.shape[0], T = args.T, A1 = adj, feat_layers= args.mlpX,  drop_non_residual = args.dropout, drop_residual = args.dropout,  criterion = 'log_softmax', w_relu = True,  w_relu_A = False).to(device)
        elif args.model_type == 'AGPCN-LINK':
            model  = AGPCN_LINK(nclass=num_labels, n_features=features.shape[1], nc_hidden= args.hidden, num_nodes = features.shape[0], T = args.T, A1 = adj, feat_layers= args.mlpX, drop_non_residual = args.dropout, drop_residual = args.dropout,  criterion = 'log_softmax', w_relu = True,  w_relu_A = False).to(device)
        else:
            print('Unspcified model type....')
            return
        
        optimizer = optim.Adam(model.parameters(),  lr=args.lr, weight_decay=args.weight_decay)

        bad_counter = 0
        best = 999999999
        for epoch in range(args.epochs):
            loss_tra,acc_tra = train_step(model,optimizer,features,labels,adj,idx_train)
            loss_val,acc_val = validate_step(model,features,labels,adj,idx_val)

            if(epoch+1)%50 == 0:
                print('Epoch:{:04d}'.format(epoch+1),
                    'train',
                    'loss:{:.3f}'.format(loss_tra),
                    'acc:{:.2f}'.format(acc_tra*100),
                    '| val',
                    'loss:{:.3f}'.format(loss_val),
                    'acc:{:.2f}'.format(acc_val*100))

            if loss_val < best:
                best = loss_val
                torch.save(model.state_dict(), checkpt_file)
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                break


        model.load_state_dict(torch.load(checkpt_file))
        loss_val, acc_val = validate_step(model, features, labels, adj, idx_val)
        loss_test, acc_test = test_step(model, features, labels, adj, idx_test)
        return acc_test*100, acc_val*100

    t_total = time.time()



    datastr = args.data
    splitstr = 'splits/'+args.data+'_split_0.6_0.2_'+str(j)+'.npz'
    acc_test,acc_val= train(datastr, splitstr)
    acc_list.append(acc_test)



print("Accuracy : " + str(np.mean(acc_list)) + ' (' + str(np.std(acc_list)) +')')

#filename = './full_GPCN_gridsearch_new/final/'+str(args.data) + '_' + str(args.model_type) + '_'   + '_' + str(args.seed) + ' ' + str(args.lr) + '_' + str(args.hidden)   + '_' + str(args.weight_decay) + '_'  + str(args.T)  + '_'  + str(args.mlpX)  + '_' + str(args.dropout) + ' ' + '.mat'
#sio.savemat(filename, {'res': acc_list})



    
