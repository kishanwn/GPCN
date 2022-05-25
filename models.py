#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:22:26 2021

@author: kishan
"""

import torch
import torch.nn as nn 
import numpy as np 
import torch.nn.functional as F
import math

## Non-adaptive models ------------------------------------------------------------------------------------------------------------------------------------------------------
class GPCN(nn.Module):
    def __init__(
        self,
        nclass,
        n_features, 
        nc_hidden,
        num_nodes,
        T,
        A1,
        feat_layers,
        res_scale,
        drop_non_residual = 0,
        drop_residual = 0,
        criterion = 'log_softmax',
        w_relu = False,
        w_relu_A = False
        ):
        super().__init__()
        self.feat_layers = feat_layers
        self.criterion = criterion
        self.w_relu = w_relu
        self.w_relu_A = w_relu_A
        self.drop_residual = drop_residual
        self.first_layers = nn.Linear(n_features, nc_hidden)
        
        self.weight = nn.Linear(nc_hidden, nc_hidden)
        self.layers_X = nn.ModuleList()
        for a in range(1,self.feat_layers ):
            self.layers_X.append(nn.Linear(nc_hidden, nc_hidden))  
        self.last_layers = nn.Linear(nc_hidden, nclass)

        self.dropout_non_residual = nn.Dropout(p=drop_non_residual)
        self.dropout_residual = nn.Dropout(p=drop_residual)
        self.A1 = A1
        self.T = T
        self.res_scale = res_scale

    def forward(self, x):
        out = self.dropout_non_residual(x)
        out = self.first_layers(out)
        if(self.w_relu == True):
            out = F.relu(out)
        for a in range(1,self.feat_layers ):
            out = self.dropout_non_residual(out)
            out = self.layers_X[a-1](out)
            if(self.w_relu == True):
                out = F.relu(out)
        
        for i in range(self.T):
            out = self.dropout_residual(out)
            
            if(self.w_relu_A == True):
                out = out + self.res_scale*F.relu(torch.spmm(self.A1 , self.weight(out))) 
            else:
                out = out + self.res_scale*torch.spmm(self.A1 , self.weight(out)) 
        
        out = self.dropout_non_residual(out)
        out = self.last_layers(out)

        if self.criterion == 'log_softmax':
            return F.log_softmax(out, dim=-1)
        else:
            return out
        

class GPCN_LINK(nn.Module):
    def __init__(
        self,
        nclass,
        n_features, 
        nc_hidden,
        num_nodes,
        T,
        A1,
        feat_layers,
        res_scale,
        drop_non_residual = 0,
        drop_residual = 0,
        criterion = 'log_softmax',
        w_relu = False,
        w_relu_A = False
        ):
        super().__init__()
        self.feat_layers = feat_layers
        self.criterion = criterion
        self.w_relu = w_relu
        self.w_relu_A = w_relu
        self.drop_residual = drop_residual
        self.first_layers = nn.Linear(n_features, nc_hidden)
        
        self.weight = nn.Linear(nc_hidden, nc_hidden)
        self.layers_X = nn.ModuleList()
        for a in range(1,self.feat_layers ):
            self.layers_X.append(nn.Linear(nc_hidden, nc_hidden))  
        self.last_layers = nn.Linear(nc_hidden, nclass)
        
        self.last_layerA = nn.Linear(num_nodes, nc_hidden)

        self.dropout_non_residual = nn.Dropout(p=drop_non_residual)
        self.dropout_residual = nn.Dropout(p=drop_residual)
        self.A1 = A1
        self.T = T
        self.res_scale = res_scale
        
        self.scaler_param = nn.Parameter(torch.randn(2, 1 ))
        self.scaler = 0

    def forward(self, x):
        out = self.dropout_non_residual(x)
        out = self.first_layers(out)
        if(self.w_relu == True):
            out = F.relu(out)
        for a in range(1,self.feat_layers ):
            out = self.dropout_non_residual(out)
            out = self.layers_X[a-1](out)
            if(self.w_relu == True):
                out = F.relu(out)
        
        for i in range(self.T):
            out = self.dropout_residual(out)
            
            if(self.w_relu_A == True):
                out = out + self.res_scale*F.relu(torch.spmm(self.A1 , self.weight(out))) 
            else:
                out = out + self.res_scale*torch.spmm(self.A1 , self.weight(out)) 
        
        self.scaler = torch.softmax(self.scaler_param,dim=0)
        out = (self.scaler[0])*out + (self.scaler[1])*self.last_layerA(self.A1)
        out = self.dropout_non_residual(out)
        out = self.last_layers(out)

        if self.criterion == 'log_softmax':
            return F.log_softmax(out, dim=-1)
        else:
            return out


## Adatpive models ------------------------------------------------------------------------------------------------------------------------------------------------------------
        
class AGPCN(nn.Module):
    def __init__(
        self,
        nclass,
        n_features, 
        nc_hidden,
        num_nodes,
        T,
        A1,
        feat_layers,
        drop_non_residual = 0,
        drop_residual = 0,
        criterion = 'log_softmax',
        w_relu = True,
        w_relu_A = False
        ):
        super().__init__()
        self.feat_layers = feat_layers
        self.criterion = criterion
        self.w_relu = w_relu
        self.w_relu_A = w_relu
        self.drop_residual = drop_residual
        self.first_layers = nn.Linear(n_features, nc_hidden)
        
        self.weight = nn.Linear(nc_hidden, nc_hidden)
        self.layers_X = nn.ModuleList()
        for a in range(1,self.feat_layers ):
            self.layers_X.append(nn.Linear(nc_hidden, nc_hidden))  
        self.last_layers = nn.Linear(nc_hidden, nclass)
 
        self.dropout_non_residual = nn.Dropout(p=drop_non_residual)
        self.dropout_residual = nn.Dropout(p=drop_residual)
        self.A1 = A1
        self.T = T
        
        self.scaler_T_param = nn.Parameter(torch.randn(T, 1 ))
        self.scaler_feat_layers = 0
        self.scaler_T = 0
        
    def forward(self, x):
        out = self.dropout_non_residual(x)
        out = self.first_layers(out)

        if(self.w_relu == True):
            out = F.relu(out)
        for a in range(1,self.feat_layers ):
            out = self.dropout_non_residual(out)
            out = self.layers_X[a-1](out)
            if(self.w_relu == True):
                out = F.relu(out)
                
      
        for i in range(self.T):
            out = self.dropout_residual(out)
            
            if(self.w_relu_A == True):
                out = out + self.scaler_T_param[i]*F.relu(torch.spmm(self.A1 , self.weight(out))) 
            else:
                out = out + self.scaler_T_param[i]*torch.spmm(self.A1 , self.weight(out)) 
        
        out = self.dropout_non_residual(out)
        out = self.last_layers(out)

        if self.criterion == 'log_softmax':
            return F.log_softmax(out, dim=-1)
        else:
            return out
        
class AGPCN_LINK(nn.Module):
    def __init__(
        self,
        nclass,
        n_features, #nc in original paper
        nc_hidden,
        num_nodes,
        T,
        A1,
        feat_layers,
        drop_non_residual = 0,
        drop_residual = 0,
        criterion = 'log_softmax',
        w_relu = True,
        w_relu_A = False
        ):
        super().__init__()
        self.feat_layers = feat_layers
        self.criterion = criterion
        self.w_relu = w_relu
        self.w_relu_A = w_relu
        self.drop_residual = drop_residual
        self.first_layers = nn.Linear(n_features, nc_hidden)
        
        self.weight = nn.Linear(nc_hidden, nc_hidden)
        self.layers_X = nn.ModuleList()
        for a in range(1,self.feat_layers ):
            self.layers_X.append(nn.Linear(nc_hidden, nc_hidden))  
        self.last_layers = nn.Linear(nc_hidden, nclass)
        
        self.last_layerA = nn.Linear(num_nodes, nc_hidden)

        self.dropout_non_residual = nn.Dropout(p=drop_non_residual)
        self.dropout_residual = nn.Dropout(p=drop_residual)
        self.A1 = A1
        self.T = T

        
        self.scaler_param = nn.Parameter(torch.randn(2, 1 ))
        self.scaler_T_param = nn.Parameter(torch.randn(T, 1 ))
        self.scaler = 0
        self.scaler_feat_layers = 0
        self.scaler_T = 0
        
    def forward(self, x):
        out = self.dropout_non_residual(x)
        out = self.first_layers(out)

        if(self.w_relu == True):
            out = F.relu(out)
        for a in range(1,self.feat_layers ):
            out = self.dropout_non_residual(out)
            out = self.layers_X[a-1](out)
            if(self.w_relu == True):
                out = F.relu(out)
                
       
        for i in range(self.T):
            out = self.dropout_residual(out)
            
            if(self.w_relu_A == True):
                out = out + self.scaler_T_param[i]*F.relu(torch.spmm(self.A1 , self.weight(out))) 
            else:
                out = out + self.scaler_T_param[i]*torch.spmm(self.A1 , self.weight(out)) 
        
        self.scaler = torch.softmax(self.scaler_param,dim=0)
        #print(self.scaler)
        out = (self.scaler[0])*out + (self.scaler[1])*self.last_layerA(self.A1)
        out = self.dropout_non_residual(out)
        out = self.last_layers(out)

        if self.criterion == 'log_softmax':
            return F.log_softmax(out, dim=-1)
        else:
            return out