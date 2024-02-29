#Import the necessary libraries and algorithm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
image = plt.imread("path_to_image")
# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
X = image.reshape(-1,3)
kmeans = KMeans(n_clusters=2,n_init=10)
kmeans.fit(X)
segemented_img = kmeans.cluster_centers_[kmeans.labels_]
segemented_img = segemented_img.reshape(image.shape)
plt.imshow(segemented_img / 255)
plt.show()