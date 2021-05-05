import LDA
import KNN
import cv2
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

testData = []
trainingSet = []
testingSet = []

for i in range(1, 41):
    im2 = cv2.imread('./faces/' + str(i) + '.pgm', 0)
    img_col = np.array(im2, dtype='float64').flatten()
    testData.append(img_col)

for i in range(1, 41):
    im2 = cv2.imread('./nonfaces/' + str(i) + '.jpg', 0)
    img_col = np.array(im2, dtype='float64').flatten()
    testData.append(img_col)
test_training_labels = np.repeat(np.arange(1, 3), 20)

n = 40
for i in range(0, 2 * n):
    if i % 2 != 0:
        trainingSet.append(testData[i])
    else:
        testingSet.append(testData[i])

trainingSet = np.array(trainingSet)
testingSet = np.array(testingSet)
testData = np.array(testData)

clf = KNN.KNN(1)
lda = LDA.LDA(1)
lda.fit(trainingSet, test_training_labels[0:n])
projected_training_data = lda.transform(trainingSet)
projected_testing_data = lda.transform(testingSet)
clf.fit(projected_training_data, test_training_labels[0:n])
predictions = clf.predict(projected_testing_data)
accuracy = np.sum(
    test_training_labels[0:n] == predictions[0:n]) / len(test_training_labels[0:n]) * 100
face_accuracy = np.sum(
    test_training_labels[0:20] == predictions[0:20]) / len(test_training_labels[0:20]) * 100
nonface_accuracy = np.sum(
    test_training_labels[20:n] == predictions[20:n]) / len(test_training_labels[20:n]) * 100
print("Using LDA with k-nearest neighbour of value 1 using ",
      (n - 20) / 20 * 100, "% of the nonfaces dataset")
print("    - total overall accuracy is ", accuracy, "%")
print("    - accuracy of non-faces: ", nonface_accuracy, "%")
print("    - accuracy of faces: ", face_accuracy, "%")
