import sys
import matplotlib.pyplot as plt
import numpy as np

#This file takes in the desired and predicted, and puts them on top of eachother to see.

def plotPrices():
	def read_file(filename):
		print("Reading: " , filename)
		fo = open(filename,'r')
		name = filename
		a1 = [float(x) for x in fo]
		print a1
		return a1

	desired = read_file("desiredList_noVol.csv")
	predicted = read_file("outputList_noVol.csv")

	legend = ['Desired','Predicted']
	plt.plot(desired)
	plt.plot(predicted)
	plt.legend(legend)
	plt.xlabel('TimeStep in day')
	plt.ylabel('Price')
	plt.title("V1 Single Layer LSTM Price Prediction every second, without Volume")
	plt.show()

def plotDiffs():
	def read_file(filename):
		print("Reading: " , filename)
		fo = open(filename,'r')
		name = filename
		a1 = np.array([float(x) for x in fo])
		print a1
		return a1

	desired = read_file("desiredList_noVol.csv")
	predicted = read_file("outputList_noVol.csv")

	myGraph = list(predicted-desired)

	
	plt.plot(myGraph)
	plt.xlabel('TimeStep in day')
	plt.ylabel('Price')
	plt.title("Residuals - V1 Single Layer LSTM Price Prediction every second, without Volume")
	plt.show()

plotDiffs()







# def read_file(filename):
# 	print('Reading: ' , filename)
# 	fo = open(filename,'r')
# 	line1 = fo.readline()
# 	line2 = fo.readline()
# 	line3 = fo.readline()
# 	name = str(fo.readline())
# 	a1 = [float(x) for x in fo.readline()[:-2].split(',')]
# 	p1 = [float(x) for x in fo.readline()[:-2].split(',')]
# 	return name,a1, p1	


# legend = []

# name,a1,p1 = read_file("results/Dielectric_Corrected_TiO2/test_out_file_2100.txt")
# legend.append(name + "_actual")
# legend.append(name+"_predicted")

# plt.plot(a1)
# plt.plot(p1)