"""
Created on Sun Feb 24 09:49:27 2019

@author: Karim Karimov, Colorado State University
"""

import torch
import random
from torch import nn
import torch.autograd as autograd




''' Below themode parameter should be a list with names of applied configurations.
------------------------------------------------------------------------------
Possible configurations: ['V','NIG', 'NFG', 'NOG', 'CIFG', 'NP', 'FGR','NIAF','NOAF']
V - default configuration from the article
NIG - No input gate: i_t = 1
NFG - No forget gate: f_t = 1
NOG - No output gate: o_t = 1
CIFG - Coupled input and forget gate: f_t = 1 − i_t
NP - No peepholes
FGR - Fullgate recurrence
NIAF - No input activation function: g(x) = x (only with other modes)
NOAF - No output activation function: h(x) = x (only with other modes)
------------------------------------------------------------------------------
Example: mode = ['FRG', 'NOAF']
'''


'''
------------------------------------------------------------------------------
Define initial inputs
------------------------------------------------------------------------------
'''
def inputs(hidden_dim = 88, input_dim = 88, input_x = False):
    random.seed(1)
    torch.manual_seed(1)
    # if run wtih CUDA: torch.cuda.is_available(): torch.cuda.manual_seed_all(1)
    s0 = torch.zeros(hidden_dim, 1)
    f0 = torch.zeros(hidden_dim, 1)
    o0 = torch.zeros(hidden_dim, 1)
    y0 = torch.rand(hidden_dim, 1) / hidden_dim
    c0 = torch.zeros(hidden_dim, 1)
    if not input_x:
        x1 = torch.zeros(input_dim, 1)
    else:
        x1 = torch.randn(input_dim, 1)
    return s0, f0, o0, y0, c0, x1

''' 
---------------------------------------------------------------------------
Initialize weights:
------------------------------------------------------------------------------
'''
def initialize(mode = ['V'], hidden_dim = 88, input_dim = 88, weights_scale = 5): 
    # hidden_dim number of nodes in a memory cell (dimension of memory cell input and outout)
    # input_m dimension if input if any
    # weight_scale measure how many time the standard deviation is less then  1.
    random.seed(2)
    torch.manual_seed(2)
    # if run wtih CUDA: torch.cuda.is_available(): torch.cuda.manual_seed_all(2)
    W_z = (torch.randn(hidden_dim, input_dim)/weights_scale).requires_grad_()
    W_s = (torch.randn(hidden_dim, input_dim)/weights_scale).requires_grad_()
    W_f = (torch.randn(hidden_dim, input_dim)/weights_scale).requires_grad_()
    W_o = (torch.randn(hidden_dim, input_dim)/weights_scale).requires_grad_()
    
    R_z = (torch.randn(hidden_dim, hidden_dim)/weights_scale).requires_grad_()
    R_s = (torch.randn(hidden_dim, hidden_dim)/weights_scale).requires_grad_()
    R_f = (torch.randn(hidden_dim, hidden_dim)/weights_scale).requires_grad_()
    R_o = (torch.randn(hidden_dim, hidden_dim)/weights_scale).requires_grad_()
    
    b_z = (torch.rand(hidden_dim, 1) - .5).requires_grad_()
    b_s = (torch.rand(hidden_dim, 1) - .5).requires_grad_()
    b_f = (torch.rand(hidden_dim, 1) - .5).requires_grad_()
    b_o = (torch.rand(hidden_dim, 1) - .5).requires_grad_()
    
    if 'NIG' in mode:
        p_f = (torch.randn(hidden_dim, 1)/weights_scale).requires_grad_()
        p_o = (torch.randn(hidden_dim, 1)/weights_scale).requires_grad_()
        weights = [W_z, W_f, W_o, R_z, R_f, R_o, b_z, b_f, b_o, p_f, p_o]
        
    elif 'NFG' in mode:
        p_s = (torch.randn(hidden_dim, 1)/weights_scale).requires_grad_()
        p_o = (torch.randn(hidden_dim, 1)/weights_scale).requires_grad_()
        weights = [W_z, W_s, W_o, R_z, R_s, R_o, b_z, b_s, b_o, p_s, p_o]
        
    elif 'NOG' in mode:
        p_s = (torch.randn(hidden_dim, 1)/weights_scale).requires_grad_()
        p_f = (torch.randn(hidden_dim, 1)/weights_scale).requires_grad_()
        weights = [W_z, W_s, W_f, R_z, R_s, R_f, b_z, b_s, b_f, p_s, p_f]

    elif 'CIFG' in mode:
        p_s = (torch.randn(hidden_dim, 1)/weights_scale).requires_grad_()
        p_o = (torch.randn(hidden_dim, 1)/weights_scale).requires_grad_()
        weights = [W_z, W_s, W_o, R_z, R_s, R_o, b_z, b_s, b_o, p_s, p_o]
    
    elif 'NP' in mode:     
        weights = [W_z, W_s, W_f, W_o, R_z, R_s, R_f, R_o, b_z, b_s, b_f, b_o]
    
    elif 'FGR' in mode:   
        R_ss = (torch.randn(hidden_dim, hidden_dim)/weights_scale).requires_grad_()
        R_fs = (torch.randn(hidden_dim, hidden_dim)/weights_scale).requires_grad_()
        R_os = (torch.randn(hidden_dim, hidden_dim)/weights_scale).requires_grad_()
        R_sf = (torch.randn(hidden_dim, hidden_dim)/weights_scale).requires_grad_()
        R_ff = (torch.randn(hidden_dim, hidden_dim)/weights_scale).requires_grad_()
        R_of = (torch.randn(hidden_dim, hidden_dim)/weights_scale).requires_grad_()
        R_so = (torch.randn(hidden_dim, hidden_dim)/weights_scale).requires_grad_()
        R_fo = (torch.randn(hidden_dim, hidden_dim)/weights_scale).requires_grad_()
        R_oo = (torch.randn(hidden_dim, hidden_dim)/weights_scale).requires_grad_()
        p_s = (torch.randn(hidden_dim, 1)/weights_scale).requires_grad_()
        p_f = (torch.randn(hidden_dim, 1)/weights_scale).requires_grad_()
        p_o = (torch.randn(hidden_dim, 1)/weights_scale).requires_grad_()
        weights = [W_z, W_s, W_f, W_o, R_z, R_s, R_f, R_o, b_z, b_s, b_f, b_o, 
                   p_s, p_f, p_o, R_ss, R_fs, R_os, R_sf, R_ff, R_of, R_so, R_fo, R_oo]
        
    else:    
        p_s = (torch.randn(hidden_dim, 1)/weights_scale).requires_grad_()
        p_f = (torch.randn(hidden_dim, 1)/weights_scale).requires_grad_()
        p_o = (torch.randn(hidden_dim, 1)/weights_scale).requires_grad_()
        weights = [W_z, W_s, W_f, W_o, R_z, R_s, R_f, R_o, b_z, b_s, b_f, b_o,
                   p_s, p_f, p_o]
    return weights


'''
------------------------------------------------------------------------------
Forward pass
------------------------------------------------------------------------------
'''

def Forward_one(mode, y0, c0, weights, x1, number_of_blocks, hidden_dim = 88, input_dim = 88, input_x = False):
    
    for _ in range(number_of_blocks):
       
        if 'V' in mode:
            z1 = torch.mm(weights[0], x1) + torch.mm(weights[4], y0) + weights[8]
            if 'NIAF' not in mode:
                z1 = z1.tanh()
            s1 = torch.mm(weights[1], x1) + torch.mm(weights[5], y0) + weights[9] + torch.mul(weights[12], c0)
            f1 = torch.mm(weights[2], x1) + torch.mm(weights[6], y0) + weights[10] + torch.mul(weights[13], c0)
            s1, f1 =  s1.sigmoid(), f1.sigmoid()
            c1 = torch.mul(z1, s1) + torch.mul(c0, f1)
            o1 = torch.mm(weights[3], x1) + torch.mm(weights[7], y0) + weights[11] + torch.mul(weights[14], c1)
        
        if 'NIG' in mode:
            z1 = torch.mm(weights[0], x1) + torch.mm(weights[3], y0) + weights[6]
            s1 = torch.ones(hidden_dim, 1)
            f1 = torch.mm(weights[1], x1) + torch.mm(weights[4], y0) + weights[7] + torch.mul(weights[9], c0)
            if 'NIAF' not in mode:
                z1 = z1.tanh()
            f1 = f1.sigmoid()
            c1 = torch.mul(z1, s1) + torch.mul(c0, f1)
            o1 = torch.mm(weights[2], x1) + torch.mm(weights[5], y0) + weights[8] + torch.mul(weights[10], c1)
            
        if 'NFG' in mode:
            z1 = torch.mm(weights[0], x1) + torch.mm(weights[3], y0) + weights[6]
            f1 = torch.ones(hidden_dim, 1)
            s1 = torch.mm(weights[1], x1) + torch.mm(weights[4], y0) + weights[7] + torch.mul(weights[9], c0)
            if 'NIAF' not in mode:
                z1 = z1.tanh()
            s1 = s1.sigmoid()
            c1 = torch.mul(z1, s1) + torch.mul(c0, f1)
            o1 = torch.mm(weights[2], x1) + torch.mm(weights[5], y0) + weights[8] + torch.mul(weights[10], c1)

        if 'NOG' in mode:
            z1 = torch.mm(weights[0], x1) + torch.mm(weights[3], y0) + weights[6]
            s1 = torch.mm(weights[1], x1) + torch.mm(weights[4], y0) + weights[7] + torch.mul(weights[9], c0)
            f1 = torch.mm(weights[2], x1) + torch.mm(weights[5], y0) + weights[8] + torch.mul(weights[10], c0)
            if 'NIAF' not in mode:
                z1 = z1.tanh()
            s1, f1 = s1.sigmoid(), f1.sigmoid()
            c1 = torch.mul(z1, s1) + torch.mul(c0, f1)
            o1 = torch.ones(hidden_dim, 1)
        
        if 'CIFG' in mode:
            z1 = torch.mm(weights[0], x1) + torch.mm(weights[3], y0) + weights[6]
            s1 = torch.mm(weights[1], x1) + torch.mm(weights[4], y0) + weights[7] + torch.mul(weights[9], c0)
            if 'NIAF' not in mode:
                z1 = z1.tanh()
            s1 = s1.sigmoid()
            f1 = torch.ones(hidden_dim, 1) - s1
            c1 = torch.mul(z1, s1) + torch.mul(c0, f1)
            o1 = torch.mm(weights[2], x1) + torch.mm(weights[5], y0) + weights[8] + torch.mul(weights[10], c1)
        
        if 'NP' in mode:
            z1 = torch.mm(weights[0], x1) + torch.mm(weights[4], y0) + weights[8]
            if 'NIAF' not in mode:
                z1 = z1.tanh()
            s1 = torch.mm(weights[1], x1) + torch.mm(weights[5], y0) + weights[9]
            f1 = torch.mm(weights[2], x1) + torch.mm(weights[6], y0) + weights[10]
            o1 = torch.mm(weights[3], x1) + torch.mm(weights[7], y0) + weights[11]
            s1, f1, o1 = s1.sigmoid(), f1.sigmoid(), o1.sigmoid()
            c1 = torch.mul(z1, s1) + torch.mul(c0, f1)
        
        if 'FGR' in mode: 
            s0 = inputs(hidden_dim, input_dim, input_x)[0]
            f0 = inputs(hidden_dim, input_dim, input_x)[1]
            o0 = inputs(hidden_dim, input_dim, input_x)[2]
            z1 = torch.mm(weights[0], x1) + torch.mm(weights[4], y0) + weights[8]
            if 'NIAF' not in mode:
                z1 = z1.tanh()
            s1 = torch.mm(weights[1], x1) + torch.mm(weights[5], y0) + weights[9] + torch.mul(weights[12], c0) + torch.mm(weights[15], s0) + torch.mm(weights[16], f0) + torch.mm(weights[17], o0)
            f1 = torch.mm(weights[2], x1) + torch.mm(weights[6], y0) + weights[10] + torch.mul(weights[13], c0) + torch.mm(weights[18], s0) + torch.mm(weights[19], f0) + torch.mm(weights[20], o0)
            c1 = torch.mul(z1, s1) + torch.mul(c0, f1)
            o1 = torch.mm(weights[3], x1) + torch.mm(weights[7], y0) + weights[11] + torch.mul(weights[14], c0) + torch.mm(weights[21], s0) + torch.mm(weights[22], f0) + torch.mm(weights[23], o0)
            s1, f1, o1 =  s1.sigmoid(), f1.sigmoid(), o1.sigmoid()
                        
        if 'NOAF' not in mode:
            c1 = c1.tanh()
            
        y1 = torch.mul(c1, o1)
#        with torch.no_grad():
#            c0 = c1
        
        return y1, c1

'''
Loop for many modes
'''
def Train_modes(data = train, modes = modes, epoch_number = 1, hidden_dim = 88, input_dim = 88, number_of_blocks = 1, weights_scale = 5, learning_rate = 2, beta = 1, input_x = False, need_losses = True):
    batch_number = len(data)
    y_train = Input_data(data)
    lengths_train = Length_input_data(data)
    weights_train = list()
    LOSS_modes = list()
    LOSS_vectors = list()
    for mode in modes:
        weights = initialize(mode = mode, hidden_dim = hidden_dim, input_dim = input_dim, weights_scale = weights_scale)
        if not input_x:
            x1 = inputs(hidden_dim, input_dim, input_x)[-1]

        for epoch in range(epoch_number):
            if epoch > 0:
                
                weights = weights1
#            LOSS_VECTOR = list()   
            if epoch == epoch_number -1:
                LOSS_VECTOR_buffer = list()   
            for batch in range(batch_number):
                
                c0 = inputs(hidden_dim, input_dim, input_x)[-2]
                
                if batch > 0:
                    weights = weights1
                
                for i in range(lengths_train[batch])[:-1]:
                    c0 = inputs()[-2]
                    y0 = y_train[batch][:, i: i + 1]
                    y_target = y_train[batch][:, i+1]
                    if i > 1:
                        weights = weights1
                        with torch.no_grad():
                            Delta0 = Delta
                    y1 = Forward_one(mode, y0, c0, weights, x1, number_of_blocks, hidden_dim, input_dim, input_x)[0]
                    loss = NLL(y_target, y1[:,0])
#                    print(loss)
                    Delta = autograd.grad(loss, inputs = weights)
                    weights1 = list()
                    for j in range(len(weights)):
                        if i > 1:
                            with torch.no_grad():
                                weights_buffer = weights[j] - learning_rate * (beta * Delta[j] +  (1 - beta) * Delta0[j])
                        else:
                            with torch.no_grad():
                                weights_buffer = weights[j] - learning_rate * Delta[j]
                        weights1.append(weights_buffer.requires_grad_())
#                    if epoch == epoch_number - 1:
                    if i == 0 and batch == 0:
#                        loss_vector = loss.data.cpu().numpy()
                        LOSS_mode = loss.data.cpu().numpy()

                    else:
#                        loss_vector = np.vstack((loss_vector, loss.data.cpu().numpy()))
                        LOSS_mode = np.vstack((LOSS_mode, loss.data.cpu().numpy()))
                    
                    if epoch == epoch_number -1:
                        if i == 0:
                            loss_buffer = loss.data.cpu().numpy()
                        else:
                            loss_buffer = np.vstack((loss_buffer, loss.data.cpu().numpy()))
                
                if epoch == epoch_number -1:
                    LOSS_VECTOR_buffer.append(loss_buffer)
        
        #        print('Batch: ' + str(batch) + ' - Loss: ' + str(loss_vector.mean()))
#            EPOCH_LOSS_MEAN = sum([row.mean() / len(LOSS_VECTOR) for row in LOSS_VECTOR])
#            if epoch == epoch_number-1:
            EPOCH_LOSS_MEAN = LOSS_mode.sum() / LOSS_mode.shape[0]
            print('Mode: ' + str(mode) + ' - Epoch: ' + str(epoch +1) + ' - Loss_mean: ' + str(EPOCH_LOSS_MEAN) + ' - Loss_median: ' + str(np.median(LOSS_mode)))
   
         
        weights_train.append(weights1)
        LOSS_modes.append(LOSS_mode)
        LOSS_vectors.append(LOSS_VECTOR_buffer)
    return weights_train, LOSS_modes, LOSS_vectors

'''
-------------------------------------------------------------------------------
Fit Data
-------------------------------------------------------------------------------
'''
def Fit_modes(data, modes, weights, hidden_dim = 88, input_dim = 88, number_of_blocks = 1, input_x = False, need_losses = True):
    batch_number = len(data)
    y_train = Input_data(data)
    lengths_train = Length_input_data(data)
    LOSS_modes = list()
    LOSS_vectors = list()
    for k in range(len(modes)):

        if not input_x:
            x1 = inputs(hidden_dim, input_dim, input_x)[-1]

            LOSS_VECTOR_buffer = list()   

            for batch in range(batch_number):
                
                c0 = inputs(hidden_dim, input_dim, input_x)[-2]
                
                for i in range(lengths_train[batch])[:-1]:
                    c0 = inputs()[-2]
                    y0 = y_train[batch][:, i: i + 1]
                    y_target = y_train[batch][:, i+1]
                    y1 = Forward_one(modes[k], y0, c0, weights[k], x1, number_of_blocks, hidden_dim, input_dim, input_x)[0]
                    loss = NLL(y_target, y1[:,0])
                    if i == 0:
                        loss_buffer = loss.data.cpu().numpy()
                        if batch == 0:
                            LOSS_mode = loss.data.cpu().numpy()

                    else:
                        loss_buffer = np.vstack((loss_buffer, loss.data.cpu().numpy()))
                        LOSS_mode = np.vstack((LOSS_mode, loss.data.cpu().numpy()))
                
                LOSS_VECTOR_buffer.append(loss_buffer)
        
            EPOCH_LOSS_MEAN = LOSS_mode.sum() / LOSS_mode.shape[0]
            print('Mode: ' + str(modes[k]) + ' - Loss_mean: ' + str(EPOCH_LOSS_MEAN) + ' - Loss_median: ' + str(np.median(LOSS_mode)))
   
         
        LOSS_modes.append(LOSS_mode)
        LOSS_vectors.append(LOSS_VECTOR_buffer)
    
    return LOSS_modes, LOSS_vectors

''' 
------------------------------------------------------------------------------
Negative log-likelihood function is define as error. It should be maximized
i.e. minimize entropy, the reason why there is a negative sign in the output)
------------------------------------------------------------------------------
'''
def NLL(target, predicted, dim = 0):
    Soft = nn.Softmax(dim = dim) # define target distribution (dim depends on same as dim of input martix), can change to target / number of non zero elements actually in our case
    LogSoft = nn.LogSoftmax(dim = dim) # define log of predicted distribution
    return - torch.mul(Soft(target), LogSoft(predicted)).sum() # find negative log-likelihood
#    return torch.mul(y_target / hidden_dim, LogSoft(predicted)).sum()

'''
------------------------------------------------------------------------------
Ayxillary routines
------------------------------------------------------------------------------
'''
def Length_input_data(data):
    
    return [len(row.T) for row in data]

def Input_data(data):
    
    return [torch.from_numpy(row).float() for row in data]

