import numpy as np

def nearest_neighbors(labels, features):
    predicted_labels = []
    for i, s in enumerate(features):
        sample = features[i]
        mask = np.ones(len(features),dtype=bool)
        mask[i] = 0
        neighbors = features[mask]
        prediction = find_nearest_neighbors(sample, neighbors, labels[mask])
        predicted_labels.append(prediction)
    score = (labels == np.array(predicted_labels)).sum()
    score = score / len(features)
    return score

def find_nearest_neighbors(sample, neighbors, labels):
    diff = (sample-np.array(neighbors)) ** 2
    if len(diff.shape) != 1:
        diff = diff.sum(axis = 1)
    min_idx = np.argmin(diff)
    neigh_label = labels[min_idx]
    return neigh_label