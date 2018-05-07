from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import create_database as cd
import affinityPropagationClustering as myaff

centers = [[1, 1], [-1, -1], [1, -1]]
data, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)
#print(data)
#data = cd.generateTData(100)
#data = data[:,[0,2]]

# Compute Affinity Propagation
af = AffinityPropagation(preference=-50).fit(data)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Scikit Results')
print(labels)
print(cluster_centers_indices)


(c,exemplars) = myaff.affinityPropagationR2(data,myaff.inverseEuclid,lam=0.5,maxiter=100)

#print(data)
print('My Results')
print(c)
print(exemplars)
#nclusters = np.max(c)
#for i in range(nclusters+1):
#    plt.plot(data[c==i,1],data[c==i,2],'ro')

#plt.show()  

