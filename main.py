import numpy as np
import matplotlib.pyplot as plt
from generation_func import *
from Two_HL_net import *
from convolution_net import *

# create data generator
data_generator = Generation_Func()

# train_[0] : inputs , train_[1] : labels
# linear
train_linear = data_generator.generation_linear()
test_linear = data_generator.generation_linear()

# xor
train_XOR = data_generator.generation_XOR_easy()
test_XOR = data_generator.generation_XOR_easy()

# training epoch
Epoch = 100000

# show learning curve
def Learning_curve(loss1, loss2):
    line1, = plt.plot([n for n in range(1,Epoch+1)], loss1, label="Train")
    line2, = plt.plot([n for n in range(1,Epoch+1)], loss2, label="Test")
    plt.legend([line1, line2], ["Train error", "Test error"], loc='upper right')
    plt.title('Learning curve')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

# show result
def show_reslut(x, y, pred_y):
    # visualize
    print(pred_y)
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    pred_list = np.zeros(y.shape)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] > 0.9:
            plt.plot(x[i][0], x[i][1], 'bo')
            pred_list[i] = 1
        else:
            plt.plot(x[i][0], x[i][1], 'ro')
    
    # Accuracy of label 0 & 1
    pred = pred_list.T.tolist()[0]
    truth = y.T.tolist()[0]
    correct = [0,0]
    for i in range(len(pred)):
        if int(pred[i]) == int(truth[i]):
            if int(truth[i]) == 0:
                correct[0] += int(1)
            else:
                correct[1] += int(1)

    Acc = [correct[0]/truth.count(0), correct[1]/truth.count(1)]
    print('Accuracy rate : [ 0 : %.1f percent , 1 : %.1f percent]' %(Acc[0]*100, Acc[1]*100))
    
    plt.show()

# trainer
def trainer(net, train_data, test_data):
    print('Start training ...')
    Loss_train = []
    Loss_test = []
    for epoch in range(Epoch):
        net.mode = 'train'
        net.forward(train_data[0])
        loss_train = net.backward(train_data[0], train_data[1])
        Loss_train.append(loss_train)

        net.mode = 'eval'
        net.forward(test_data[0])
        loss_test = net.backward(test_data[0], test_data[1])
        Loss_test.append(loss_test)
   
        if (epoch+1)%(Epoch/10) == 0:
            print('Epoch : %d || Train Loss : %.10f || Test Loss : %.10f' %(epoch+1, loss_train, loss_test))

    print('Finish training')
    Learning_curve(Loss_train, Loss_test)

# Two hidden layers

# XOR
net_xor = two_hl_net()
trainer(net_xor, train_XOR, test_XOR)
# Make prediction
pred = net_xor.forward(test_XOR[0])
show_reslut(test_XOR[0], test_XOR[1], pred)

# #-----------------------------------------------------#
# # Linear
# net_linear = two_hl_net(n=3,act='tanh',lr=0.01)
net_linear = two_hl_net()
trainer(net_linear, train_linear, test_linear)
# Make prediction
pred = net_linear.forward(test_linear[0])
show_reslut(test_linear[0], test_linear[1], pred)

#-----------------------------------------------------#
# Convolution
# XOR
# net_conv_xor = Conv()
# trainer(net_conv_xor, train_XOR, test_XOR)
# # Make prediction
# pred = net_conv_xor.forward(test_XOR[0])
# show_reslut(test_XOR[0], test_XOR[1], pred)

# nett_conv_xor = Conv(M=3)
# trainer(nett_conv_xor, train_XOR, test_XOR)
# # Make prediction
# pred = nett_conv_xor.forward(test_XOR[0])
# show_reslut(test_XOR[0], test_XOR[1], pred)
# #-----------------------------------------------------#
# # Linear
# net_conv_linear = Conv()
# trainer(net_conv_linear, train_linear, test_linear)
# # Make prediction
# pred = net_conv_linear.forward(test_linear[0])
# show_reslut(test_linear[0], test_linear[1], pred)

# nett_conv_linear = Conv(M=3)
# trainer(nett_conv_linear, train_linear, test_linear)
# # Make prediction
# pred = nett_conv_linear.forward(test_linear[0])
# show_reslut(test_linear[0], test_linear[1], pred)