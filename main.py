import imagehash
from PIL import Image
from glob import glob
from scipy.spatial import distance_matrix
from itertools import combinations
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN

def make_hashes(paths, hash_method=imagehash.whash):
    img_names, img_hashes = [], []
    for path in paths:
        image = Image.open(path)
        img_hash = hash_method(image)
        img_hashes.append(img_hash)
        img_names.append(path)
    return img_names, img_hashes

def get_file_paths(_dir):
    return glob(_dir)

def compute_dists(hashes):
    mat = np.zeros((len(hashes), len(hashes)))
    for i, j in combinations(range(len(hashes)), 2):
        dist = hashes[i] - hashes[j]
        mat[i, j] = mat[j,i] = dist
    return mat

def cluster(mat, fnames, eps, min_samples):
    m = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = m.fit_predict(mat)
    clusters = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[lbl].append(fnames[i])
    return clusters


dirname = './data/tmp/*'
print('Reading Files...')
paths = get_file_paths(dirname)
print('Computing Hashes...')
paths, img_hashes = make_hashes(paths)
print('Computing Distances...')
dist_mat = compute_dists(img_hashes)
print('Clustering...')
clusters = cluster(dist_mat, paths, 20, 2)
print(clusters)