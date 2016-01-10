import random as rand 
import datetime as dt
import random
import numpy as np
from PIL import Image
from math import log, exp, acos, degrees, sqrt

# define constants
global RADIUS_INIT
RADIUS_INIT = 1000
global LR_INIT
LR_INIT = 0.1

class SOMnode(object):
	# constructor
	def __init__(self, x = 0, y = 0, dim = 3):
		# set the node's central coordinates in the lattice
		self.coord = []
		self.coord.append(x)
		self.coord.append(y)
		self.dim = dim
		self.weights = []
		self.dist2BMU = 10000
		# initialize the weights
		for w in range(self.dim):
			self.weights.append(rand.random()) # this is in [0,1]!!!
		#print self.weights
		# set the boundaries of the cell for visualizaton purposes
		self.left  = self.coord[0] - 1 
		self.right = self.coord[0] + 1
		self.top   = self.coord[1] - 1
		self.bottom= self.coord[1] + 1
	def getDistance(self,inputVector,method="euclidean"):
		if method == "cosine":
			# l2 normalize!!!!
			distance = 0
			for iterator in range(len(inputVector)):
				distance += (self.weights[iterator]*inputVector[iterator])

			return degrees(acos(distance)) 
		else:		
			distance = 0
			for iterator in range(len(inputVector)):
				distance += (self.weights[iterator]-inputVector[iterator])**2
				#print self.weights[iterator], inputVector[iterator]
			#print distance**0.5
			return  distance**(0.5)
	def euclideanDist(self,targetCoord):
		distance = 0
		for iterator in range(len(targetCoord)):
			distance += (self.coord[iterator]-targetCoord[iterator])**2
			#print self.weights[iterator], inputVector[iterator]
		#print distance**0.5
		return distance**(0.5)		
	def updateWeights(self, inputVector, learningRate):
		for iterator in range(len(self.weights)):
			self.weights[iterator] = self.weights[iterator] \
									 + self.dist2BMU * learningRate * \
									 (inputVector[iterator]\
									  - self.weights[iterator]) 
	def setDist2BMU(self, dist2BMU):
		self.dist2BMU = dist2BMU								 


class SOMlattice(object):
	def __init__(self, iterNo, lr=0.1):
		# members
		self.LAMBDA  = RADIUS_INIT / log(iterNo)
		self.sNodes  = []
		self.lr      = lr
		self.BMU_id  = 0
		self.inputVector = [] # at time t
		self.neighbors = []
		self.currentRadius = 1000
	def setInputVector(self, inVec):
		self.inputVector = inVec	
	def calculateCurrentRadius(self, timeT):
		# use exponential decay function
		return RADIUS_INIT * exp(-(float(timeT) / self.LAMBDA))
	def addSomNode(self, SOMnodeX):
		self.sNodes.append(SOMnodeX)
	def getBestMatchingUnit(self):
		bestMatch = 10000
		for nodeIdx in range(len(self.sNodes)):
			matchRate = self.sNodes[nodeIdx].getDistance(self.inputVector)
			if matchRate < bestMatch:
				bestMatch = matchRate
				BMU_id = nodeIdx
		# update the index of the BMU		
		self.BMU_id = BMU_id
	def getNeighbors(self,timeT):
		currentRadius = self.currentRadius
		# create a list
		neighbors_tmp = []
		for nodeIdx in range(len(self.sNodes)):
			dist2BMU = self.sNodes[nodeIdx].euclideanDist(self.sNodes[self.BMU_id].coord)
			if dist2BMU < currentRadius:
				neighbors_tmp.append(nodeIdx)
				# update node's distance to BMU
				#self.sNodes[nodeIdx].setDist2BMU(dist2BMU)
				self.sNodes[nodeIdx].setDist2BMU(self.distSNode2BMU(self.sNodes[nodeIdx],timeT))
		self.neighbors = neighbors_tmp				
	def updateLearningRate(self,timeT):
		self.lr = LR_INIT  *  exp(-(float(timeT) / self.LAMBDA))
	def updateCurrentRadius(self,timeT):
		self.currentRadius = self.calculateCurrentRadius(timeT)	
	def updateNeighborWeights(self):
		for idx in range(len(self.neighbors)):
			self.sNodes[idx].updateWeights(self.inputVector,self.lr)	
	def distSNode2BMU(self, SOMnodeX, timeT):
		dist2BMU = SOMnodeX.euclideanDist(self.sNodes[self.BMU_id].coord)
		return exp(-(dist2BMU**2 / (2*self.calculateCurrentRadius(timeT))))
	def initializeNetwork(self, config, data):
		print str(dt.datetime.now()) + ": Network is being initialized."
		for nodeId in range(len(config)):
		  tmpSomNode = SOMnode(config[nodeId][0], config[nodeId][1], len(data[0]))
		  self.addSomNode(tmpSomNode)
		  print str(dt.datetime.now()) + ": Node initialized @ (%f, %f)." %(config[nodeId][0],config[nodeId][1])
		print str(dt.datetime.now()) + ": Network is ready."
	def visualize(self, cellWidth,epoch):
		lattice = np.zeros((sqrt(len(self.sNodes))*cellWidth,sqrt(len(self.sNodes))*cellWidth,3), dtype = np.uint8)
		node = 0
		lastY = 0
		for ii in range(int(sqrt(len(self.sNodes)))):
			lastX = 0
			for jj in range(int(sqrt(len(self.sNodes)))):
				lattice[lastX:lastX+cellWidth,lastY:lastY+cellWidth,0] += np.floor(255.0*self.sNodes[node].weights[0]) 
				lattice[lastX:lastX+cellWidth,lastY:lastY+cellWidth,1] += np.floor(255.0*self.sNodes[node].weights[1])
				lattice[lastX:lastX+cellWidth,lastY:lastY+cellWidth,2] += np.floor(255.0*self.sNodes[node].weights[2])
				lastX += cellWidth
				node += 1
			lastY += cellWidth
		img = Image.fromarray(lattice, 'RGB')
		img.save("screenshot@epoch-"+str(epoch)+".jpg")
		return lattice


class Config(object):
	def __init__(self):
		self.coords = []
		self.latticeSize = [0, 0]

	def readConfig(self, configFile):
		print str(dt.datetime.now()) + ": Network configuration is being read."
		# read data coordinates
		iterator = 0
		with open(configFile,'r') as configuration:
			for coords in configuration:
				tmp 		= coords.split(" ")
				if iterator == 0:
					self.latticeSize[0] = int(tmp[0])
				elif iterator == 1:
					self.latticeSize[1] = int(tmp[0])
				else:	
					self.coords.append([float(x) for x in tmp])
				iterator += 1 
		# error handling	
		try:
			assert(self.latticeSize[0] * self.latticeSize[1] == len(self.coords))
		except AssertionError:
			print str(dt.datetime.now()) + ": Error: Lattice size mismatch!"
		else:
			print str(dt.datetime.now()) + ": Configuration has been succesfully read."	 	

class Data(object):
	def __init__(self):
		self.vecs = []

	def readData(self, dataFile):
		# read data coordinates
		print str(dt.datetime.now()) + ": Data is being read."
		with open(dataFile,'r') as data:
			for sample in data:
				tmp = sample.split(" ")
				self.vecs.append([float(x) for x in tmp])
		print str(dt.datetime.now()) + ": Data has been succesfully read."	 	
# creates a dataset and a configuration file for testing
def createDatasets(latticeSize,sampleSize):
    with open("test_config-small.txt","w") as txt:
		txt.write(str(latticeSize) + "\n")
		txt.write(str(latticeSize) + "\n")
		for x in range(latticeSize):
	   	   for y in range(latticeSize):
	       	       txt.write(str(x+1) + " " + str(y+1) +"\n")
    with open("test_data-small.txt","w") as txt:
		for x in range(sampleSize):
		    txt.write(str(rand.random()) + " " + str(rand.random()) + " "+ str(rand.random())+"\n")

if __name__ == "__main__":
    # create test samples
    latticeSize = 400
    sampleSize = 200  
    createDatasets(latticeSize,sampleSize)
    # read config file
    config       = Config()
    config.readConfig("test_config-small.txt")
    configParam  = config.latticeSize
    configCoord  = config.coords 
    # read data 
    data = Data()
    data.readData("test_data-small.txt")
    dataVecs = data.vecs
    # create a network object
    sampleNo = len(dataVecs) # equal to number of samples
    SOMnet = SOMlattice(sampleNo)
    # initialize it
    SOMnet.initializeNetwork(configCoord, dataVecs)
    # training params
    EPOCH  = 4
    epoch  = 0
    # training loop
    print str(dt.datetime.now()) + ": Starting the training.."
    # set running index
    timeT = 0
    while epoch != EPOCH:
		# randomize order
		random.shuffle(dataVecs)
		for sample in dataVecs:			
			SOMnet.updateCurrentRadius(timeT)
			# set current input vector
			SOMnet.setInputVector(sample)
			# get the best matching unit and its neighbors			
			SOMnet.getBestMatchingUnit()
			SOMnet.getNeighbors(timeT)
			# update the neural weights of the nodes within the 
			# neighborhood
			SOMnet.updateNeighborWeights()
			# update learning rate
			SOMnet.updateLearningRate(epoch)
			print str(dt.datetime.now()) + ": Epoch: %d, Iter: %d, radius: %f, lr: %f" %(epoch, timeT, SOMnet.currentRadius, SOMnet.lr)
			timeT += 1	
		# visualize the lattice	
		a = SOMnet.visualize(10,epoch)	
		epoch += 1	
