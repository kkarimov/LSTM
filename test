"""
Created on Sun Feb 24 09:49:27 2019

@author: Karim Karimov, Colorado State University
"""

modes = [['V'] , ['CIFG'], ['FGR'], ['NP'], ['NOG'], ['V', 'NIAF'], ['NIG'], ['NFG'], ['V', 'NOAF']]

import chorales
import LSTM_PH

## train LSTM for different modes
# it is better to start from one mode of ypur choce and one epoch

modes = [['V']]

weights_trained_buffer, losses_buffer, loss_vector_buffer = Train_modes(data = train, modes = modes, epoch_number = 1, hidden_dim = 88, input_dim = 88, number_of_blocks = 1, weights_scale = 5, learning_rate = 1, beta = .8, input_x = False, need_losses = True)

## fit LSTM to validation data set for mode of your choice

losses, losses_vector = Fit_modes(valid, modes, weights_trained_buffer, hidden_dim = 88, input_dim = 88, number_of_blocks = 1, input_x = False, need_losses = True)

