import imagehash
from PIL import Image
from glob import glob
from scipy.spatial import distance_matrix
from itertools import combinations
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
import cv2 as cv
import matplotlib.pyplot as plt 

def make_hashes(paths, hash_method=imagehash.whash):
    img_names, img_hashes = [], []
    for path in paths:
        print(path)
        
        image = Image.open(path)
        height_img, width_img = int(0.1*float(image.size[0])), int(0.1*float(image.size[1]))
        image = image.resize((width_img, height_img))
        
        img_hash = hash_method(image)
        img_hashes.append(img_hash)
        img_names.append(path)
    return img_names, img_hashes

def get_file_paths(_dir):
    return glob(_dir)

def resize_images(images, factor=0.1):
    resized_images = []
    for img in images:
        height_img, width_img = int(0.1*float(img.shape[0])), int(0.1*float(img.shape[1]))
        resized_img = cv.resize(img, (width_img, height_img), interpolation=cv.INTER_AREA)
        resized_images.append(resized_img)
    return resized_images

def makeSIFTdistance(img1, img2):
    sift = cv.xfeatures2d.SIFT_create()
    kp_sift_1, sift_des_1 = sift.detectAndCompute(img1, None)
    kp_sift_2, sift_des_2 = sift.detectAndCompute(img2, None)

    bf_matcher = cv.BFMatcher()
    matches = bf_matcher.knnMatch(sift_des_1, sift_des_2, k=2)

    count = 0
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            count += 1
    return count

def compute_hash_dists(hashes):
    mat = np.zeros((len(hashes), len(hashes)))
    for i, j in combinations(range(len(hashes)), 2):
        dist = hashes[i] - hashes[j]
        mat[i, j] = mat[j,i] = dist
    return mat

def compute_sift_dists(fnames):
    images = []
    for fname in fnames:
        images.append(cv.imread(fname, 0))
    print('Resizing Images...')
    images = resize_images(images, 0.1)
    print('Computing SIFT common features matrix..')
    mat = np.zeros((len(fnames), len(fnames)))
    for i, j in combinations(range(len(images)), 2):
        img1, img2 = images[i], images[j]
        dist = makeSIFTdistance(img1, img2)
        mat[i, j] = mat[j, i] = dist
    mat = np.exp(-mat)
    return mat

def cluster(mat, fnames, eps, min_samples):
    m = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = m.fit_predict(mat)
    clusters = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[lbl].append(fnames[i])
    return clusters

def clusterMain(dirname, option=0):
    print('Reading Files...')
    paths = get_file_paths(dirname)

    if option is 0:
        print('Computing Hashes...')
        paths, img_hashes = make_hashes(paths)
        print('Computing Distances...')
        dist_mat = compute_hash_dists(img_hashes)
        print('Clustering...')
        clusters = cluster(dist_mat, paths, 18, 1)
    if option is 1:
        dist_mat = compute_sift_dists(paths)
        print('Clustering...')
        clusters = cluster(dist_mat, paths, 0.1, 1)
    return clusters


def testSIFT(img1, img2):
    img1, img2 = cv.imread(img1, 0), cv.imread(img2, 0)
    # img1_gray, img2_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY), cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    
    height_img1, width_img1 = int(0.1*float(img1.shape[0])), int(0.1*float(img1.shape[1]))
    height_img2, width_img2 = int(0.1*float(img2.shape[0])), int(0.1*float(img2.shape[1]))

    resized_img1 = cv.resize(img1, (width_img1, height_img1), interpolation=cv.INTER_AREA)
    resized_img2 = cv.resize(img2, (width_img2, height_img2), interpolation=cv.INTER_AREA)
    
    # orb = cv.ORB_create()
    # kp_orb_1, orb_des_1 = orb.detectAndCompute(resized_img1, None)
    # kp_orb_2, orb_des_2 = orb.detectAndCompute(resized_img2, None)

    # #Brute Force Matcher
    # bf_matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # # Match descriptors
    # matches = bf_matcher.match(orb_des_1, orb_des_2)
    # matches = sorted(matches, key = lambda x:x.distance)
    # print(matches[0].distance)
    # # Draw first 10 matches.
    # img3 = cv.drawMatches(resized_img1, kp_orb_1, resized_img2, kp_orb_2, matches[:10], None, flags=2)


    sift = cv.xfeatures2d.SIFT_create()
    kp_sift_1, sift_des_1 = sift.detectAndCompute(resized_img1, None)
    kp_sift_2, sift_des_2 = sift.detectAndCompute(resized_img2, None)

    bf_matcher = cv.BFMatcher()
    matches = bf_matcher.knnMatch(sift_des_1, sift_des_2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append([m])

    img3 = cv.drawMatchesKnn(resized_img1, kp_sift_1 ,resized_img2, kp_sift_2, good, None,  flags=2)

    plt.imshow(img3) 
    plt.show()

clusters = clusterMain('./data/tmp/*', 0)
print(clusters)