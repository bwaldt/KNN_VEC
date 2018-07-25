import numpy as np
def knn_test(K, trainPoints, testPoints, distFunc='l2'):
    '''
KNN_TEST - Evaluates KNN predictions given training data and parameters.
 
  [testLabels] = knn_test(K, trainPoints, trainLabels, testPoints, ...
                          [distFunc])

   K - Number of nearest neighbors to use
   trainPoints - N x P matrix of examples, where N = number of points and
       P = dimensionality
   trainLabels - N x 1 vector of labels for each training point.
   testPoints  - M x P matrix of examples, where M = number of test points
       and P = dimensionality
   distFunc - OPTIONAL string declaring which distance function to use:
       valid functions are 'l2','l1', and 'linf'

   Returns sorted distances to the nearest neighbors and index values of those neighbors
             '''
        


    # % NOTE: this code is heavily VECTORIZED, which means that it does not use a
    # % any "for" loops and runs very quickly

    numTestPoints  =  testPoints.shape[0]
    numTrainPoints = trainPoints.shape[0]

    # % The following lines compute the difference between every test point and
    # % every train point in each dimension separately, using a single M x P X N
    # % 3-D array subtraction:

    # % Step 1:  Reshape the N x P training matrix into a 1 X P x N 3-D array
    trainMat = trainPoints.T.reshape((1, trainPoints.shape[1], numTrainPoints))
    
    # % Step 2:  Replicate the training array for each test point (1st dim)
    trainCompareMat = np.tile(trainMat, [numTestPoints, 1, 1])
                     
    # % Step 3:  Replicate the test array for each training point (3rd dim)
    N = testPoints[:,:,np.newaxis]
    testCompareMat = np.tile(N, [1, 1, numTrainPoints])
                                 
    # % Step 4:  Element-wise subtraction
    diffMat = testCompareMat - trainCompareMat;

    # % Now we can compute the distance functions on these element-wise
    # % differences:
    if distFunc=='l2':
        distMat = np.sqrt(np.sum(diffMat**2, axis = 1));
    elif distFunc=='l1':
        distMat = np.sum(np.absolute(diffMat), axis = 1);
    elif distFunc=='linf':
        distMat = max(np.absolute(diffMat), [], 2);
    else:
        error('Unrecognized distance function');
    

#     % Now we have a M x 1 x N 3-D array of distances between each pair of points.
#     % To find the nearest neighbors, we first "squeeze" this to become a M x N
#     % matrix, and then sort within each of the M rows separately. Note that we
#     % use only the second output from the "sort" command.

    distMat = np.squeeze(distMat);
    if numTestPoints == 1: #% if only 1 point, squeeze converts to col vector
        distMat = distMat.T
    
    idxSorted    = np.argsort(distMat, axis= 1)
    sortedDistMat= np.sort(distMat, axis= 1)


    return sortedDistMat,idxSorted
