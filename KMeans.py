from argparse import ArgumentParser
import numpy as np

class KMeans:
    def __init__(self, k, train, test):
        self.k = k
        self.classes = np.loadtxt(train, delimiter=',', usecols=[0], dtype='int')
        self.mostFrequentClass = np.empty(self.k, dtype='int')
        self.data = np.loadtxt(train, delimiter=',')
        # remove classes - don't want to cheat :)        
        self.data = self.data[:,1:]
        self.nData = self.data[:,0].size
        self.nFeatures = self.data[0,:].size
        # k centers, each with n features
        self.centers = np.empty((self.k,self.nFeatures))
        
    def run(self):
        self.initClusters()
        while not self.learn():
            pass
        self.labelCenters()
        
    def initClusters(self):
        # arrays of mins and maxs for each feature
        mins = np.amin(self.data, axis=0)
        maxs = np.amax(self.data, axis=0)
        
        # init each center with random values in the range of the features
        for i in range(self.k):
            # mins + (array of rands(0,1))*ranges
            self.centers[i,:] = mins + np.random.random(self.nFeatures)* \
                                (maxs - mins)
        
    def learn(self):
        """Compute distances, find the closest cluster, and update
        the cluster centers"""

        # remeber the old centers for comparison
        oldCenters = np.copy(self.centers)

        # compute distances
        distances = np.ones((1,self.nData))*np.sum((self.data-self.centers[0,:])**2,axis=1)
        for i in xrange(self.k - 1):
            distances = np.append(distances, np.ones((1,self.nData))*np.sum((self.data - self.centers[i + 1,:])**2, axis=1), axis=0)

        # find the closest cluster
        self.cluster = distances.argmin(axis=0)
        self.cluster = np.transpose(self.cluster*np.ones((1,self.nData), dtype='int'))

        # update the cluster centers
        for i in range(self.k):
            thisCluster = np.where(self.cluster == i, 1, 0)
            if np.sum(thisCluster) > 0:
                self.centers[i,:] = np.sum(self.data*thisCluster, axis=0)/np.sum(thisCluster)

        # elements of centerDiff are true if oldCenters == newCenters
        # and false if they are not
        centerDiff = np.in1d(oldCenters, self.centers)
        nUnequal = centerDiff[np.where(centerDiff == False)].size
        # if no centers have changed position, return true to indicate
        # that learning is complete
        return nUnequal == 0

    def labelCenters(self):
        for i in range(self.k):
            thisCluster = np.where(self.cluster == i, 1, 0)
            clusterClasses = np.transpose(thisCluster)*self.classes
            clusterClasses = clusterClasses[0,:]
            counts = np.bincount(clusterClasses)
            # find the bin with the highest count, excluding 0
            self.mostFrequentClass[i] = np.argmax(counts[1:]) + 1

        print self.mostFrequentClass
parser = ArgumentParser()
parser.add_argument('-k', type=int, default=3, help='number of clusters')
args = parser.parse_args()
kMeans = KMeans(args.k, 'data/wine.train', 'data/wine.test')
kMeans.run()

