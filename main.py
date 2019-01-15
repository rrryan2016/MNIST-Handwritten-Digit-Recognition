# As usual, a bit of setup
import numpy as np
import matplotlib.pyplot as plt

from support.classifiers.cnn import *
from data_load import get_MNIST_data

from support.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from support.layers import *
from support.fast_layers import *
from support.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Data Loading from MNIST
data = get_MNIST_data()
for k, v in data.items():
  print('%s: ' % k, v.shape)

    ######################################################################################
    #                             Test of overfit small data                             #
    ######################################################################################
#
# # Test of overfit small data
#
# num_train = 100
#
# small_data = {
#   'X_train': data['X_train'][:num_train],
#   'y_train': data['y_train'][:num_train],
#   'X_val': data['X_val'],
#   'y_val': data['y_val'],
#     }
#
# # # Show the input data
# # for i in range(10):
# #     print(small_data['y_train'][i])
# #     plt.imshow(small_data['X_train'][i][0],cmap='gray')
# #     plt.pause(0.000001)
# #     plt.show()
#
# model = ThreeLayerConvNet(weight_scale=1e-2)
#
# solver = Solver(model, small_data,
#                 num_epochs=20, batch_size=50,
#                 update_rule='adam',
#                 optim_config={
#                   'learning_rate': 1e-4,
#                 },
#                 verbose=True, print_every=1)
# solver.train()
#
# plt.subplot(2, 1, 1)
# plt.plot(solver.loss_history, 'o')
# plt.xlabel('iteration')
# plt.ylabel('loss')
#
# plt.subplot(2, 1, 2)
# plt.plot(solver.train_acc_history, '-o')
# plt.plot(solver.val_acc_history, '-o')
# plt.legend(['train', 'val'], loc='upper left')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()

    ######################################################################################
    #                           End Test of overfit small data                           #
    ######################################################################################

    #################################################################################
    #                             Test of train the net                             #
    #################################################################################


# Test of Train the net

model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

solver = Solver(model, data,
                num_epochs=1, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-4,
                },
                verbose=True, print_every=100)
solver.train()
print(solver.best_val_acc)

    #####################################################################################
    #                             End Test of train the net                             #
    #####################################################################################



# Initialization of Cross Validation
num_folds = 5

X_train_folds = []
y_train_folds = []

X_train_folds = np.split(data['X_train'],num_folds)
y_train_folds = np.split(data['y_train'],num_folds) # List Object


    ###########################################################################################
    #                             Cross Validation of filter_size                             #
    ###########################################################################################

# print('\n Cross Validation of filter_size: \n')
# k_choices = [1,3,5,7]
# k_val_acc = {}
# for k in k_choices:
#   print('Calculating: k = ',k)
#   k_val_acc[k] = np.zeros(num_folds)

#   X_train = data['X_train']
#   y_train = data['y_train']


#   for i in range(num_folds):
#     print('Cross Validation Processing ',i+1 ,' / ',num_folds)
#     Xtr = np.vstack((np.array(X_train_folds)[:i],np.array(X_train_folds)[(i+1):]))
#     ytr = np.vstack((np.array(y_train_folds)[:i],np.array(y_train_folds)[(i+1):]))
#     Xte = np.array(X_train_folds)[i]
#     yte = np.array(y_train_folds)[i]

#     a,b,c,d,e = Xtr.shape
#     Xtr.resize(a*b,c,d,e)
#     a,b = ytr.shape
#     ytr.resize(a*b)

#     new_data = {'X_train':Xtr,'y_train':ytr,'X_val':Xte,'y_val':yte}

#     model = ThreeLayerConvNet(filter_size=k, weight_scale=0.001, hidden_dim=500, reg=0.001)
#     solver = Solver(model, new_data,
#                     num_epochs=1, batch_size=50,
#                     update_rule='adam',
#                     optim_config={
#                       'learning_rate': 1e-4,
#                     },
#                   verbose=True, print_every=300)
#     solver.train()

#     k_val_acc[k][i] = solver.best_val_acc

# # print out the computed val_acc
# for k in sorted(k_val_acc):
#   for accuracy in k_val_acc[k]:
#     print ('k=%d, accuracy=%f' %(k,accuracy))

    ###########################################################################################
    #                          End of Cross Validation of filter_size                         #
    ###########################################################################################

    ##########################################################################################
    #                             Cross Validation of batch_size                             #
    ##########################################################################################

# print('\n Cross Validation: batch_size: \n')
# batch_size_choices = [30,50,100,200]
# batch_size_val_acc = {}
# for batchSize in batch_size_choices:
#   print('Calculating: batch_size = ',batchSize)
#   batch_size_val_acc[batchSize] = np.zeros(num_folds)

#   X_train = data['X_train']
#   y_train = data['y_train']

#   for i in range(num_folds):
#     print('Cross Validation Processing ',i+1 ,' / ',num_folds)
#     Xtr = np.vstack((np.array(X_train_folds)[:i],np.array(X_train_folds)[(i+1):]))
#     ytr = np.vstack((np.array(y_train_folds)[:i],np.array(y_train_folds)[(i+1):]))
#     Xte = np.array(X_train_folds)[i]
#     yte = np.array(y_train_folds)[i]
    
#     a,b,c,d,e = Xtr.shape
#     Xtr.resize(a*b,c,d,e)
#     a,b = ytr.shape
#     ytr.resize(a*b)

#     new_data = {'X_train':Xtr,'y_train':ytr,'X_val':Xte,'y_val':yte}

#     model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)
#     solver = Solver(model, new_data,
#                     num_epochs=1, batch_size=batchSize,
#                     update_rule='adam',
#                     optim_config={
#                       'learning_rate': 1e-4,
#                     },
#                   verbose=True, print_every=300)
#     solver.train()

#     batch_size_val_acc[batchSize][i] = solver.best_val_acc

# for k in sorted(batch_size_val_acc):
#   for accuracy in batch_size_val_acc[k]:
#     print ('batch_size=%d, accuracy=%f' %(k,accuracy))

    ###########################################################################################
    #                          End of Cross Validation of batch_size                          #
    ###########################################################################################

    ###########################################################################################
    #                             Cross Validation of Learning Rate                           #
    ###########################################################################################

# print('\n Cross Validation: learning_rate: \n')
# learning_rate_choices = [1e-1,1e-2,1e-4,1e-8,1e-16]
# learning_rate_val_acc = {}
# for learningRate in learning_rate_choices:
#   print('Calculating: learning_rate = ',learningRate)
#   learning_rate_val_acc[learningRate] = np.zeros(num_folds)

#   X_train = data['X_train']
#   y_train = data['y_train']

#   for i in range(num_folds):
#     print('Cross Validation Processing ',i+1 ,' / ',num_folds)
#     Xtr = np.vstack((np.array(X_train_folds)[:i],np.array(X_train_folds)[(i+1):]))
#     ytr = np.vstack((np.array(y_train_folds)[:i],np.array(y_train_folds)[(i+1):]))
#     Xte = np.array(X_train_folds)[i]
#     yte = np.array(y_train_folds)[i]

#     a,b,c,d,e = Xtr.shape
#     Xtr.resize(a*b,c,d,e)
#     a,b = ytr.shape
#     ytr.resize(a*b)

#     new_data = {'X_train':Xtr,'y_train':ytr,'X_val':Xte,'y_val':yte}

#     model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)
#     solver = Solver(model, new_data,
#                     num_epochs=1, batch_size=50,
#                     update_rule='adam',
#                     optim_config={
#                       'learning_rate': learningRate,
#                     },
#                   verbose=True, print_every=300)
#     solver.train()

#     learning_rate_val_acc[learningRate][i] = solver.best_val_acc

# for k in sorted(learning_rate_val_acc):
#   for accuracy in learning_rate_val_acc[k]:
#     print ('learning_rate=%d, accuracy=%f' %(k,accuracy))


    ###########################################################################################
    #                         End of Cross Validation of Learning Rate                        #
    ###########################################################################################