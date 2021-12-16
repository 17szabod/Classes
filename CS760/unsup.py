# Implements unsupervised machine learning tools on data coming from Gerry
import numpy as np
from matplotlib import pyplot as plt
import gerry
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# Collect data
def collect_samples(data, shape):
    # Annoyingly traversal order is not automated yet, so I hav to input it manually
    face_order = [[46, 48, 130], [46, 130, 48, 128, 40], [46, 116, 47, 50], [46, 47, 116], [46, 117, 47], [46, 40, 117],
                  [117, 40, 47], [128, 41, 40], [41, 131, 47, 40], [131, 52, 50, 47], [50, 52, 51], [51, 52, 106],
                  [52, 13, 106], [11, 13, 52, 22], [11, 22, 12], [22, 52, 23], [23, 52, 21], [21, 52, 131, 97],
                  [22, 23, 21], [12, 22, 35, 104], [22, 21, 35], [35, 21, 97], [132, 128, 48], [48, 127, 132],
                  [127, 128, 132], [41, 128, 45], [136, 128, 127], [126, 128, 136], [136, 127, 48], [122, 124, 136],
                  [136, 129, 126], [136, 124, 126, 129], [126, 45, 128], [126, 44, 45], [124, 44, 126],
                  [122, 43, 139, 124], [139, 43, 141, 124], [141, 43, 124], [43, 120, 124], [120, 44, 124],
                  [120, 42, 44], [44, 42, 45], [42, 41, 45], [42, 121, 131, 41], [42, 131, 121], [120, 131, 42],
                  [43, 131, 120], [122, 131, 43], [122, 136, 48], [122, 48, 134], [164, 122, 134, 48], [164, 131, 122],
                  [164, 97, 131], [164, 98, 97], [98, 86, 84, 97], [84, 35, 97], [83, 35, 84], [85, 83, 84],
                  [86, 85, 84], [151, 83, 85, 88], [86, 88, 85], [82, 35, 83], [81, 35, 82], [82, 83, 53], [81, 82, 53],
                  [54, 81, 53], [55, 54, 53], [53, 83, 57], [57, 83, 151], [151, 88, 29], [29, 88, 86, 31],
                  [87, 31, 86], [87, 96, 31], [87, 86, 98], [164, 87, 98], [164, 96, 87], [164, 95, 96], [96, 95, 31],
                  [95, 93, 30, 31], [30, 29, 31], [93, 29, 30], [55, 53, 56], [56, 53, 57], [56, 57, 58], [58, 57, 151],
                  [56, 58, 59], [56, 59, 60, 61], [59, 58, 60], [60, 58, 67], [60, 67, 61], [67, 58, 151],
                  [67, 151, 160], [67, 160, 77], [67, 77, 160, 152], [67, 152, 160, 151, 29], [67, 29, 153],
                  [67, 153, 68], [153, 76, 68], [153, 29, 68, 76], [67, 68, 69], [68, 159, 69], [68, 29, 159],
                  [159, 29, 70], [159, 70, 69], [61, 67, 69], [61, 69, 62], [56, 61, 62], [55, 56, 62], [55, 62, 64],
                  [64, 62, 63], [64, 63, 65], [63, 62, 65], [65, 62, 144], [62, 69, 144], [65, 144, 66], [144, 69, 66],
                  [66, 69, 71], [72, 66, 71], [69, 70, 71], [164, 102, 95], [164, 103, 95, 102], [164, 89, 95, 103],
                  [89, 94, 95], [95, 94, 93], [164, 161, 89], [89, 161, 94], [161, 91, 93, 94], [91, 92, 93],
                  [92, 29, 93], [164, 90, 161], [90, 168, 161], [90, 169, 161, 168], [169, 91, 161], [90, 91, 169],
                  [33, 90, 164], [33, 165, 90], [33, 166, 90, 165], [166, 91, 90], [33, 91, 166], [33, 92, 91],
                  [33, 114, 92], [177, 29, 92, 115], [177, 115, 92], [114, 177, 92], [33, 34, 32, 114], [34, 112, 32],
                  [101, 147, 177, 114], [101, 177, 147], [170, 101, 114, 32], [37, 170, 32], [112, 37, 32],
                  [112, 36, 37], [38, 170, 37], [36, 38, 37], [39, 38, 36], [112, 39, 36], [110, 39, 112],
                  [110, 38, 39], [110, 170, 38], [146, 101, 170], [146, 178, 101], [146, 179, 101, 178],
                  [146, 180, 101, 179], [146, 154, 176, 177, 101, 180], [146, 176, 154], [146, 173, 177, 176],
                  [146, 177, 173], [155, 29, 177], [78, 29, 155], [73, 78, 155], [156, 29, 78], [73, 156, 78],
                  [79, 29, 156], [73, 79, 156], [157, 29, 79], [73, 157, 79], [80, 29, 157], [73, 80, 157],
                  [158, 29, 80], [73, 158, 80], [70, 29, 158], [73, 70, 158], [71, 70, 73], [146, 73, 155, 177],
                  [146, 75, 73], [146, 149, 75], [146, 174, 149], [174, 182, 149], [146, 182, 174], [146, 171, 182],
                  [171, 172, 182], [182, 172, 149], [172, 74, 149], [149, 74, 75], [75, 74, 73], [74, 71, 73]]
    count, X = gerry.enumerate_paths_with_order(data, shape, face_order, draw=False)
    return X


# Use PCA to see whether the problem can lie in a lower dimension
def reduce_dim(X):
    pca = PCA(svd_solver='full')
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    Y = pca.transform(X[:-1])
    y = pca.transform([X[-1]])
    return Y, y


# Just calls scikit's kmeans algorithm
def kmeans(k, X):
    km = KMeans(n_clusters=k)
    km.fit(X)
    return km


# Returns the signed distance between x and y using the hyperplane y is normal to
def signed_distance(x, y):
    return np.linalg.norm(x-y) if np.sum(x-y) > 0 else -np.linalg.norm(x-y)


if __name__ == "__main__":
    try:
        X = np.load("data/samples.npy")
    except IOError:
        X = collect_samples("data/exp2627neighb.dbf", "data/exp2627wards.shp")
        np.save("data/samples", X)
    # Delete meaningless metrics from X
    X = np.delete(X, [1, 3], 1)
    # Use PCA to visualize the 2-dimensional representation
    Y, y = reduce_dim(X)
    plt.plot(Y[:, 0], Y[:, 1], 'ro')
    plt.plot(y[:, 0], y[:, 1], 'bo')
    plt.title("2-Dimensional Reduction using PCA")
    plt.show()
    # Normalize entries of X
    normed_X = np.zeros(X.shape)
    for i in range(X.shape[1]):
        col = X[:, i]
        ma = np.max(col)
        mi = np.min(col)
        rang = ma - mi
        for j in range(X.shape[0]):
            normed_X[j, i] = (X[j, i] - mi)/rang
    # Find the k-means assignments
    km = kmeans(5, normed_X[:-1])
    y = normed_X[-1]
    assignment = km.predict([y])
    cluster = normed_X[:-1][km.labels_ == assignment]
    center = km.cluster_centers_[assignment]
    cluster_distances = [signed_distance(x, center) for x in cluster]
    # Calculate statistics within our cluster
    plt.boxplot(cluster_distances, vert=False)
    plt.plot([signed_distance(y, center)], [1], 'bx')
    plt.title("Distribution of Distances within Cluster")
    plt.show()

    exit()

