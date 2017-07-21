#THis plots the values
import sys
import matplotlib.pyplot as plt
import numpy as np
#'results/Dielectric_Massive/train_train_loss_0.txt',
loss_files=['loss.csv','loss_withVolume.csv']

#Dielectric_Order_BiasTest/train_val_loss_1_val_bias_test.txt','results/Dielectric_Order_BiasTest/train_val_loss_val_bias_test_sigmoid_nobias.txt']

lossValues = np.genfromtxt(loss_files[0],delimiter=',')
newVals = range(0,len(lossValues))
plt.plot(newVals,lossValues)

lossValues2 = np.genfromtxt(loss_files[1],delimiter=',')
newVals2 = range(0,len(lossValues2))
#plt.plot(newVals2,lossValues2)


plt.xlabel("Steps (in 10's)")
plt.ylabel("MSE Training Error - Normalized points")
plt.title("Training Error error")
plt.legend(['Values every 50'])
#plt.plot(lossValues_oldbatch)
plt.show()