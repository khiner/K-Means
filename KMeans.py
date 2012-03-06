from argparse import ArgumentParser
import numpy as np

class KMeans:
    def __init__(self, k):
        self.k = k        
        
    def train(self, trainPath):
        data = np.loadtxt(trainPath, delimiter=',')
        classes = data[:,0].astype(int)
        # remove classes - don't want to cheat :)
        featureData = data[:,1:]
        nData = featureData[:,0].size
        nFeatures = featureData[0,:].size
        # k centers, each with n features
        self.centers = np.empty((self.k, nFeatures))
        self.mostFrequentClass = np.empty(self.k, dtype='int')
        self.initClusters(featureData, nFeatures)
        while not self.learn(featureData, nData):
            pass
        self.labelCenters(classes)
        
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
        the cluster centers"""

        # remeber the old centers for comparison
        oldCenters = np.copy(self.centers)

        # compute distances
        distances = self.computeDistances(data, nData)
        
        # find the closest cluster
        self.cluster = distances.argmin(axis=0)
        self.cluster = np.transpose(self.cluster*np.ones((1,nData), dtype='int'))

        # update the cluster centers
        for i in range(self.k):
            thisCluster = np.where(self.cluster == i, 1, 0)
            if np.sum(thisCluster) > 0:
                self.centers[i,:] = np.sum(data*thisCluster, axis=0)/np.sum(thisCluster)

        centerDiff = oldCenters - self.centers
        nUnequal = centerDiff[np.where(centerDiff != 0)].size
        # if no centers have changed position, return true to indicate
        # that learning is complete
        return nUnequal == 0

    def computeDistances(self, data, nData):
        distances = np.empty((self.k, nData))
        for i in xrange(self.k):
            distances[i] = np.sum((data - self.centers[i,:])**2, axis=1)
        return distances
        
    def labelCenters(self, classes):
        for i in range(self.k):
            thisCluster = np.where(self.cluster == i, 1, 0)
            clusterClasses = (np.transpose(thisCluster)*classes)[0,:]
            counts = np.bincount(clusterClasses)
            # find the bin with the highest count, excluding 0
            self.mostFrequentClass[i] = np.argmax(counts[1:]) + 1

        print self.mostFrequentClass

    def test(self, testPath):
        data = np.loadtxt(testPath, delimiter=',')
        testClasses = data[:,0].astype(int)
        testData = data[:,1:]
        nTestData = testData[:,0].size
        distances = self.computeDistances(testData, nTestData)
        self.cluster = distances.argmin(axis=0)
        predictedClasses = self.mostFrequentClass[self.cluster]
        classDiff = testClasses - predictedClasses
        nCorrect = classDiff[np.where(classDiff == 0)].size
        cohesion = self.computeCohesion(testData)
        print "accuracy", float(nCorrect)/float(testClasses.size)

    def computeCohesion(self, data):
        # currently cohesion is across all data.  need for each cluster
        cohesion = np.subtract.outer(data, data)**2
        cohesion = np.apply_over_axes(np.sum, cohesion, [1,2,3])[:,0,0,0]
        assert cohesion.size == data[:,0].size
        return cohesion
        
parser = ArgumentParser()
parser.add_argument('-k', type=int, default=3, help='number of clusters')
args = parser.parse_args()
kMeans = KMeans(args.k)
kMeans.train('data/wine.train')
kMeans.test('data/wine.test')

