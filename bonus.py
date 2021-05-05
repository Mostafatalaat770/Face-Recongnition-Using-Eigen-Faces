import LDA
import KNN
import PCA
import cv2
import os
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

x = []
y = []

width = 92
length = 112

# making the main matrix
for i in range(1, 41):
    images = os.listdir('./att_faces/s' + str(i))
    for image in images:
        img = cv2.imread('./att_faces/s' + str(i) + "/" + image, 0)
        height1, width1 = img.shape[:2]
        img_col = np.array(img, dtype='float64').flatten()
        subject = int(i)
        x.append(img_col)
        y.append(subject)

x = np.array(x)
y = np.array(y)
# Taking Test Set and Training Set
trainingSet = []
testingSet = []
training_labels = np.repeat(np.arange(1, 41), 7)
testing_labels = np.repeat(np.arange(1, 41), 3)
for i in range(0, 400):
    trainingSet.append(x[i]) if i % 10 < 7 else testingSet.append(x[i])

trainingSet = np.array(trainingSet)
testingSet = np.array(testingSet)

# ----------------------------------------PCA-------------------------------------------
for knn in [1, 3, 5, 7]:
    clf = KNN.KNN(knn)
    print("Using PCA with k-nearest neighbour of value ", knn, ' : ')
    for alpha in [0.8, 0.85, 0.9, 0.95]:
        pca = PCA.PCA(alpha)
        pca.fit(trainingSet)
        projected_training_data = pca.transform(trainingSet)
        projected_testing_data = pca.transform(testingSet)
        clf.fit(projected_training_data, training_labels)
        predictions = clf.predict(projected_testing_data)
        accuracy = np.sum(testing_labels == predictions) / \
            len(testing_labels) * 100
        print("    - accuracy for alpha = ", alpha, " is ", accuracy, "%")

# ----------------------------------------LDA-------------------------------------------
for knn in [1, 3, 5, 7]:
    clf = KNN.KNN(knn)
    print("Using LDA with k-nearest neighbour of value ", knn, ' : ')
    lda = LDA.LDA(39)
    lda.fit(trainingSet, training_labels)
    projected_training_data = lda.transform(trainingSet)
    projected_testing_data = lda.transform(testingSet)
    clf.fit(projected_training_data, training_labels)
    predictions = clf.predict(projected_testing_data)
    accuracy = np.sum(testing_labels == predictions) / \
        len(testing_labels) * 100
    print("    - accuracy is ", accuracy, "%")
