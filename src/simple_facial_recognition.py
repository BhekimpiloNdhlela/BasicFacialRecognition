#usr/bin/python
'''
author : Bhekimpilo Ndhlela
author : 18998712
module : RW364_ComputerVision
task   : Assignment 05
since  : Monday-19-10-2018
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

TRAIN_DATA_SET_PATH = '../imgs/faces_training/{}.pgm'
TEST1_DATA_SET_PATH = '../imgs/faces_test/test1/{}.pgm'
TEST2_DATA_SET_PATH = '../imgs/faces_test/test2/{}.pgm'

IMAGE_WIDTH = 92
IMAGE_HEIGHT = 112

def display_test_data_set(path, set, R=5, C=8):
    '''
    Displays the test images specified by the path 'path' to the matplotlib plot.

    path : the path of the test images for the model.
    set  : the test image set int 1 or 2, for test set 1 or 2 respectively
    R    : optional int 5 default [the rows to be displayed]
    C    : optional int 8 default [the cols to be displayed]
    '''
    fig, ax = plt.subplots(nrows=R, ncols=C, subplot_kw={'xticks':[] ,'yticks':[]})
    if   set == 1: fig.suptitle('TESTING IMAGE DATA SET: 1')
    elif set == 2: fig.suptitle('TESTING IMAGE DATA SET: 2')
    i = 1
    for r in xrange(R):
        for c in xrange(C):
            img = plt.imread(path.format(str(i)))
            ax[r, c].set_title(str(i))
            ax[r, c].imshow(img, cmap='gray')
            i = i + 1
    plt.show()

def display_train_data_set(R=5, C=8):
    '''
    Displays the training images to the matplotlib plot.

    R    : optional int 5 default [the rows to be displayed]
    C    : optional int 8 default [the cols to be displayed]
    '''
    fig, ax = plt.subplots(nrows=R, ncols=C, subplot_kw={'xticks':[] ,'yticks':[]})
    fig.suptitle('TRAINING IMAGE DATA SET')
    i = 1
    for r in xrange(R):
        for c in xrange(C):
            img = plt.imread(TRAIN_DATA_SET_PATH.format(str(i)))
            ax[r, c].imshow(img, cmap='gray')
            ax[r, c].set_title(str(i))
            i = i + 1
    plt.show()

def display_average_face(average_face):
    '''
    Displays the average face to the matplotlib plot, this function assumes that
    average_face is a flat vector representing the average face to be displayed.

    average_face: flat, float or int np.array dims(,m) where m = pXq
    '''
    average_face = get_mXn_eigenface(average_face)
    plt.title('Average Face')
    plt.imshow(average_face, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def get_mXn_eigenface(flat):
    '''
    reshapes the m length vector into a pXq matrix for the final image to be
    displayed.

    flat: is a flat int or float np.array with dim = (, m) where m=pXq
    '''
    return np.reshape(flat[:], (IMAGE_HEIGHT, IMAGE_WIDTH))

def plot_singular_vals_of_X(S, a=40):
    '''
    visualizing function for the singular values

    S   : the singular values to be ploted
    a   : optional[4 default] the choosen alpha or alpah is the Infered
          approximate dimension alpa by regarding the singular values of X below
          some cut-off as zero.
    '''
    plt.title("Plot of the singular values of X")
    plt.ylabel("singular values")
    plt.xlabel("values of a")
    plt.plot(np.linspace(1, a, a), S[:], 'ok', label='Singular values')
    plt.plot(12, 562.5, 'Dr', label='Choosen Alpha')
    plt.grid(True, linewidth=3)
    plt.legend(loc='best')
    plt.show()

def get_average_face_vector(imgs):
    '''
    calculate the average vector or the average face vector for the training
    data set.

    imgs    : the flatenned images stored in a single X matrix
    returns : np.array of dimension/ shape (,10304)
    '''
    num_imgs = float(len(imgs[0,:]))
    return np.array([float(sum(imgs[i,:]))/num_imgs for i in xrange(10304)])

def get_img_vector_matrix_F(path=TRAIN_DATA_SET_PATH):
    '''
    function to return the X image matrix with flatenned images stacked in the
    matrix.

    return: the training image matrix with flattened training images.
    '''
    imgs_matrix = np.zeros((10304, 40), dtype=float)
    for i, img in enumerate(xrange(1, 41)):
        img = plt.imread(path.format(img)).flat
        imgs_matrix[:,i] = img[:]
    return imgs_matrix

def getUa_and_S(X):
    '''
    Finding a lower dimensional basis for the image vectors is now a matter of
    finding a basis for the column space of X. Infer an approximate dimension Alpha
    by regarding the singular values of X below some cut-off as zero.

    return : turple -> (U, S) where U is a basis for the column space of X and
                                    S singular values of of X the image matrix.
    '''
    U, S, _ = la.svd(X)
    return U[:,:12], S

def get_matrix_X(a, f):
    '''
    function responsible for building the X matrix.

    a       : the average face
    f       : the n image vectors stacked in a matrix
    return  : X
    '''
    m, n = np.shape(f)
    s = 1.0/np.sqrt(n)
    X = np.zeros((m, n), dtype=float)
    for i in xrange(n):
        X[:,i] = s*(f[:,i] - a)
    return X

def scale_Ua(eigenface, smin=0.0, smax=255.0):
    '''
    this function scales columns of U = eigenface which span the colum space of X
    in an approximate sense.

    eigenface : np.array eigenface = U the matrix that span the colum space of X
    smin      : the minimum pixel expected in an image [optional] 0.0 optional
    smax      : the maximum pixel expected in an image [optional] 255.0 optional
    return    : the scaled U where columns of U span the colum space of X in an
                approximate sense.
    '''
    minUa, maxUa = min(eigenface), max(eigenface)
    scale_this = lambda ele: smin+((ele-minUa)*(smax-smin)/(maxUa-minUa))
    return np.array([int(round(scale_this(element))) for element in eigenface])

def display_matches(matches, path, test_status, R=5, C=8):
    '''
    function responsible for displaying matches of the training restults. the red
    images where failed to be identified algorithm and the gray ones where
    successfully identified.

    matches     : the matches computed by the training algorithm.
    path        : the path to the testing data set
    test_status : the testing set int 1 or 2
    R           : rows
    C           : cols optional int
    '''
    fig, ax = plt.subplots(nrows=R, ncols=C, subplot_kw={'xticks':[] ,'yticks':[]})
    string = ' 1' if test_status == 1 else ' 2'
    fig.suptitle('TEST IMAGE DATA SET'+string)
    i = 1
    for r in xrange(R):
        for c in xrange(C):
            img = plt.imread(path.format(str(i)))
            ax[r, c].set_title(str(i))
            if i - 1 == matches[i-1]:
                ax[r, c].imshow(img, cmap='gray')
            else:
                ax[r, c].imshow(img, cmap='OrRd_r')
            i = i + 1
    plt.show()

def get_best_match(ntrain, ntest):
    '''
    function to get the best match by using a new (unknown) image which is mapped
    to its low-dim. representation, and that feature vector is matched against
    the database. We compare low-dim. representations, and pick the label of the
    closest training sample.

    ntrain  : the train image matrix
    ntest   : the test image matrix
    '''
    distance = lambda p, q: np.linalg.norm(p-q)
    matches = np.zeros(40, dtype=int)
    count = 0
    for j, test in enumerate(ntest.T):
        temp_length = []
        for train in ntrain.T:
            temp_length.append(distance(train, test))
        matches[j] = temp_length.index(min(temp_length))
        count = count + 1 if matches[j] == j else count
    return matches, str(round((count/40.0)*100)) + ' %'

def test_img_reconstruct(Ua, Y, a):
    '''
    function to reconstruct the test images before their are mapped back to the
    database.

    Ua  : the matrix that span the colum space of X
    Y   : low dimensional representation of f
    a   : the avarage face of the training dataset
    '''
    reconstructed_imgs = np.zeros((10304, 40), dtype=float)
    for i in xrange(40):
        reconstructed_imgs[:,i] = np.dot(Ua, Y[:,i]) + a
    return reconstructed_imgs

def display_test_and_encoded_img(rt_img, path=TEST1_DATA_SET_PATH):
    '''
    displays the test and the encoded test images.
    '''
    fig, ax = plt.subplots(nrows=2, ncols=5, subplot_kw={'xticks':[] ,'yticks':[]})
    fig.suptitle('Test Images with their respective reconstructed imgs')
    i = 1
    for r in xrange(2):
        for c in xrange(5):
            if i < 6:
                img = plt.imread(path.format(str(i)))
                ax[r][c].set_title('O image: ' + str(i))
                ax[r][c].imshow(img, cmap='gray')
            else:
                img = get_mXn_eigenface(rt_img[:,i-6])
                ax[r][c].set_title('R image:' + str(i-5))
                ax[r][c].imshow(img, cmap='gray')
            i = i + 1
    plt.show()

def display_eigenfaces(Ua, num_eigenfaces=6, R=2, C=3):
    '''
    distaply the eigen faces with their originals above.

    Ua              : the matrix that span the colum space of X
    num_eigenfaces  : number of eigen faces to be displaed
    R               : number of desired rows to display
    C               : number of desired cols to display
    '''
    fig, ax = plt.subplots(nrows=R, ncols=C, subplot_kw={'xticks':[] ,'yticks':[]})
    fig.suptitle('The First 6 EigenFaces')
    i = 1
    for r in xrange(R):
        for c in xrange(C):
            img = get_mXn_eigenface(scale_Ua(Ua[:,i - 1]))
            ax[r, c].set_title(str(i))
            ax[r, c].imshow(img, cmap='gray')
            i = i + 1
    plt.show()

def get_eigenface_represantation(Ua, a, test_img_path):
    '''
    function that builds the encoding or the eigenfaces

    Ua              : the matrix that span the colum space of X
    a               : the average vector of the training images.
    test_img_path   : the test data set image path.
    '''
    F = get_img_vector_matrix_F(path=test_img_path)
    Y = np.zeros((np.shape(Ua.T)[0], 40), dtype=float)
    for i in xrange(40):
        Y[:,i] = np.dot(Ua.T, (F[:,i] - a))
    return Y


if __name__ == '__main__':
    F = get_img_vector_matrix_F()
    a = get_average_face_vector(F)

    display_average_face(np.ceil(a[:]).astype(int))
    X = get_matrix_X(a, F)

    Ua, S = getUa_and_S(X)
    plot_singular_vals_of_X(S)
    display_eigenfaces(Ua, num_eigenfaces=6)

    test1_encodings = get_eigenface_represantation(Ua, a, TEST1_DATA_SET_PATH)
    reconstructed_test1_img = test_img_reconstruct(Ua, test1_encodings, a)
    display_test_and_encoded_img(reconstructed_test1_img, path=TEST1_DATA_SET_PATH)

    test2_encodings = get_eigenface_represantation(Ua, a, TEST2_DATA_SET_PATH)
    reconstructed_test2_img = test_img_reconstruct(Ua, test2_encodings, a)
    display_test_and_encoded_img(reconstructed_test2_img, path=TEST2_DATA_SET_PATH)

    train_encodings = get_eigenface_represantation(Ua, a, TRAIN_DATA_SET_PATH)

    matches1, percentage1 = get_best_match(train_encodings, test1_encodings)
    display_matches(matches1, TEST1_DATA_SET_PATH, 1)

    matches2, percentage2 = get_best_match(train_encodings, test2_encodings)
    display_matches(matches2, TEST2_DATA_SET_PATH, 2)

    print 'percentage 1 == ', percentage1, '\tpercentage 2 == ', percentage2

else:
    import sys
    sys.exit('USAGE: python simple_facial_recognition.py')
