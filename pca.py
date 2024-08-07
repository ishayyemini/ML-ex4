import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people


def plot_vector_as_image(image, h, w):
    """
    utility function to plot a vector as image.
    Args:
    image - vector of pixels
    h, w - dimesnions of original pi
    """
    plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.title("title", size=12)
    plt.show()


def get_pictures_by_name(name="Ariel Sharon"):
    """
    Given a name returns all the pictures of the person with this specific name.
    YOU CAN CHANGE THIS FUNCTION!
    THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
    """
    lfw_people = load_data()
    selected_images = []
    n_samples, h, w = lfw_people.images.shape
    target_label = list(lfw_people.target_names).index(name)
    for image, target in zip(lfw_people.images, lfw_people.target):
        if target == target_label:
            image_vector = image.reshape((h * w, 1))
            selected_images.append(image_vector)
    return selected_images, h, w


def load_data():
    # Don't change the resize factor!!!
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people


######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""


def PCA(X, k):
    """
    Compute PCA on the given matrix.

    Args:
            X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
            For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
            k - number of eigenvectors to return

    Returns:
      U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
                    of the covariance matrix.
      S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
    """
    n = X.shape[0]
    Z = X - X.mean(axis=0)
    cov_matrix = (Z.T @ Z) / n
    U, S, V = np.linalg.svd(cov_matrix)
    return U[:, :k].T, S[:k]


def b():
    images, h, w = get_pictures_by_name()
    X = np.array(images)[:, :, 0]
    U, S = PCA(X, 10)

    fig = plt.figure()
    for i in range(10):
        fig.add_subplot(2, 5, i + 1)
        plt.imshow(U[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(f"Vector {i + 1}")

    plt.show()


def c():
    images, h, w = get_pictures_by_name()
    X = np.array(images)[:, :, 0]
    k_array = [1, 5, 10, 30, 50, 100]
    L2_array = []

    for k in k_array:
        U, S = PCA(X, k)
        fig = plt.figure(figsize=(3, 7))
        plt.axis("off")
        plt.suptitle(f"k = {k}")
        curr_l2 = 0

        for i, photo_i in enumerate(np.random.choice(range(X.shape[0]), 5)):
            orig_photo = X[photo_i]
            decoded_photo = U.T @ (U @ X[photo_i])

            fig.add_subplot(5, 2, 2 * i + 1)
            if i == 0:
                plt.title("Original")
            plt.axis("off")
            plt.imshow(orig_photo.reshape((h, w)), cmap=plt.cm.gray)

            fig.add_subplot(5, 2, 2 * i + 2)
            if i == 0:
                plt.title("Transformed")
            plt.axis("off")
            plt.imshow(decoded_photo.reshape((h, w)), cmap=plt.cm.gray)

            curr_l2 += np.linalg.norm(orig_photo - decoded_photo)

        L2_array.append(curr_l2)
        plt.axis("off")
        plt.show()

    plt.plot(k_array, L2_array)
    plt.xlabel("k")
    plt.ylabel("L2 distances sum")
    plt.show()


b()
c()
