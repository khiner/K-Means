"""Data should be in the format <class, <features>>, where class is an
integer, and features are comma-separated floats"""
from argparse import ArgumentParser
import numpy as np

class KMeans:
    def __init__(self, k):
        self.k = k        
        
    def train(self, trainPath):
        """
        Train using the data in the file argument.
        1) load training data, extract classes and features
        2) normalize features
        3) learn until improvement stops
        4) assosiate centroids with their cluster's majority class
        5) find avg cohesion and avg separation of training data
        """
        data = np.loadtxt(trainPath, delimiter=',')
        classes = data[:,0].astype(int)
        # remove classes - don't want to cheat :)
        featureData = data[:,1:]
        nData = featureData[:,0].size
        nFeatures = featureData[0,:].size
        # normalize features        
        featureData /= np.max(featureData, axis=0)
        # init k centers, each with n features
        self.centers = np.empty((self.k, nFeatures))
        self.mostFrequentClass = np.empty(self.k, dtype='int')
        self.initClusters(featureData, nFeatures)
        while not self.learn(featureData, nData):
            pass
        self.labelCenters(classes)
        cohesion = self.cohesion(data)
        separation = self.separation(data)        
        print 'Cohesion:', cohesion
        print 'Separation:', separation
        
    def test(self, testPath):
        """
        Test the trained centroids using the data in the file argument
        1) load test data, extract classes and features
        2) normalize features
        3) find closes centroids for each example to find cluster
        4) get predicted classes using 'mostFrequentClass' map from training
           phase
        5) compute accuracy
        """
        data = np.loadtxt(testPath, delimiter=',')
        testClasses = data[:,0].astype(int)
        testData = data[:,1:]
        nTestData = testData[:,0].size
        nFeatures = testData[0,:].size
        # normalize features
        testData /= np.max(testData, axis=0)
        # find distances from all points to all centroids
        distances = self.computeDistances(testData, nTestData)
        # find the closest centroids
        self.cluster = distances.argmin(axis=0)
        # to get predicted classes,
        # translate the cluster each example belongs to into the most
        # frequent class that cluster contained in the training data
        predictedClasses = self.mostFrequentClass[self.cluster]
        # compute accuracy
        classDiff = testClasses - predictedClasses
        nCorrect = classDiff[np.where(classDiff == 0)].size
        print "accuracy", float(nCorrect)/float(testClasses.size)
        
    def initClusters(self, data, nFeatures):
        # arrays of mins and maxs for each feature
        mins = np.amin(data, axis=0)
        maxs = np.amax(data, axis=0)
        
        # init each center with random values in the range of the features
        for i in range(self.k):
            # mins + (array of rands(0,1))*ranges
            self.centers[i,:] = mins + np.random.random(nFeatures)*(maxs - mins)
        
    def learn(self, data, nData):
        """Compute distances, find the closest cluster, and update
        the cluster centers
        Returns true if no centers have changed position, to indicate
        learning is complete"""

        # remeber the old centers for comparison
        oldCenters = np.copy(self.centers)

        # find distances from all points to all centroids
        distances = self.computeDistances(data, nData)
        
        # find the closest cluster
        self.cluster = distances.argmin(axis=0)

        # update the cluster centers
        for i in range(self.k):
            thisCluster = data[np.where(self.cluster == i)]
            clusterSize = thisCluster[:,0].size
            if clusterSize > 0:
                self.centers[i] = np.sum(thisCluster, axis=0)/clusterSize
                
        centerDiff = oldCenters - self.centers
        nUnequal = centerDiff[np.where(centerDiff != 0)].size
        # if no centers have changed position, return true to indicate
        # that learning is complete
        return nUnequal == 0

    def computeDistances(self, data, nData):
        """Returns list of sums of distances from all points to
        each centroid"""
        distance = np.empty((self.k, nData))
        for i in xrange(self.k):
            distance[i] = np.sum((data - self.centers[i,:])**2, axis=1)
        return distance
        
    def labelCenters(self, classes):
        """Associate each centroid with the majority class in its cluster"""
        for i in range(self.k):
            thisCluster = np.where(self.cluster == i, 1, 0)
            clusterClasses = (np.transpose(thisCluster)*classes)
            counts = np.bincount(clusterClasses)
            # find the bin with the highest count, excluding 0
            if counts[1:].size:
                self.mostFrequentClass[i] = np.argmax(counts[1:]) + 1
            else:
                self.mostFrequentClass[i] = 0

    def cohesion(self, data):
        """Returns avg cohesion"""
        cohesion = np.zeros(self.k)        
        
        for i in range(self.k):
            thisCluster = data[np.where(self.cluster == i)]
            # if there are no points associated with this centroid, bail
            if thisCluster.size == 0:
                continue
            thisClusterSize = thisCluster[:,0].size
            thisCohesion = np.zeros(thisClusterSize**2)
            N = 0
            for j in xrange(thisClusterSize):
                for k in xrange(j + 1, thisClusterSize):
                    thisCohesion[j*k + k] = np.sum((thisCluster[j] - thisCluster[k])**2)
                    N += 1                    
            cohesion[i] = N/np.sum(thisCohesion)
            
        # non-zero elements correspond contain all centroids with points
        return np.average(cohesion[np.nonzero(cohesion)])

    def separation(self, data):
        """Returns avg separation between each pair of clusters"""
        separation = np.zeros(self.k*self.k)
        # currently cohesion is across all data.  need for each cluster
        for i in range(self.k):
            thisCluster = data[np.where(self.cluster == i)]
            # if there are no points associated with this centroid, bail
            if thisCluster.size == 0:
                continue
            thisClusterSize = thisCluster[:,0].size
            for j in range(i + 1, self.k):
                thatCluster = data[np.where(self.cluster == j)]
                # if there are no points associated with this centroid, bail
                if thatCluster.size == 0:
                    continue 
                thatClusterSize = thatCluster[:,0].size
                thisSeparation = np.zeros(thisClusterSize*thatClusterSize)
                for m in xrange(thisClusterSize):
                    for n in xrange(thatClusterSize):
                        thisSeparation[m*n + n] = np.sum((thisCluster[m] - thatCluster[n])**2)
                separation[i*j + j] = np.sum(thisSeparation)/(thisClusterSize*thatClusterSize)

        # non-zero elements correspond contain all combinations of centroids
        # where both centroids have associated points
        return np.average(separation[np.nonzero(separation)])
        
parser = ArgumentParser()
parser.add_argument('-k', type=int, default=3, help='number of clusters')
parser.add_argument('-r', '--train', default='data/wine.train', help='train file. default is data/wine.train')
parser.add_argument('-t', '--test', default='data/wine.test', help='test file. default is data/wine.test')
args = parser.parse_args()
kMeans = KMeans(args.k)
kMeans.train(args.train)
kMeans.test(args.test)

