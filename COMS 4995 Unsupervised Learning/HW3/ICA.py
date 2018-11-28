# Import packages.
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy import linalg as LA


def preprocessing(x):
    """
    centering and whitening the data x
    :param x: input data
    :return: the data after centering and whitening
    """
    # Centering
    x[0] = x[0] - x[0].mean()
    x[1] = x[1] - x[1].mean()

    # Whitening
    # Calculate cov matrix
    cov = np.cov(x)
    # Calculate eigenvalues and eigenvectors of the covariance matrix.
    d, E = LA.eigh(cov)
    # Generate a diagonal matrix with the eigenvalues as diagonal elements.
    D = np.diag(d)
    D0 = LA.sqrtm(LA.inv(D))
    xw = np.dot(E, np.dot(D0, np.dot(np.transpose(E), x)))

    # Plot whitened data to show new structure of the data.
    plt.figure()
    plt.plot(xw[0], xw[1], '*b')
    plt.ylabel('Signal 2')
    plt.xlabel('Signal 1')
    plt.title("Whitened data")
    plt.savefig("output/Whitened data.jpg")

    return xw


def FastICA(x):
    """
    perform fastICA
    :param x: data after preprocessing
    :return: the original sources matrix
    """
    max_iter = 100000
    epsilon = 1e-4
    W = [0] * len(x)
    for temp_i in range(len(x)):
        W[temp_i] = np.random.rand(len(x), 1)
        for temp_iter in range(max_iter):
            if temp_iter == max_iter - 1:
                print("Exceeds maximum iterating number!")
            new_W = 1/len(x[0]) * (np.dot(x, np.tanh(np.dot(W[temp_i].transpose(), x)).transpose())
            - np.dot((1 - np.tanh(np.dot(W[temp_i].transpose(), x) ** 2)),
                     np.ones(len(x[0])).transpose()) * W[temp_i])
            for temp_j in range(temp_i):
                new_W = new_W - np.dot(new_W.transpose(), W[temp_j]) * W[temp_j]
            new_W = new_W / LA.norm(new_W)

            # check converge
            if np.abs(np.abs(np.dot(W[temp_i].transpose(), new_W)) - 1)[0][0] < epsilon:
                W[temp_i] = new_W
                break
            else:
                W[temp_i] = new_W
    W_matrix = np.vstack((W[0].transpose(), W[1].transpose()))

    return np.dot(W_matrix, x)


def FOBI(x):
    """
    perform Fourth Order Blind Identification
    :param x: data after preprocessing
    :return: the original sources matrix
    """

    norm_x = LA.norm(x, axis=0)
    norm = [norm_x, norm_x]
    cov = np.cov(np.multiply(norm, x))

    d, Y = LA.eigh(cov)

    source = np.dot(np.transpose(Y), x)
    return source


def ICA(x, method):
    """
    Apply two methods of ICA
    :param x: data after preprocessing
    :param method: FOBI or FastICA
    :return: source data
    """
    if method == "FOBI":
        return FOBI(x)
    elif method == "FastICA":
        return FastICA(x)
    else:
        raise ValueError("The method can only be FOBI or FastICA")


def analysis(file1, file2, method):
    """
    :param file1: the file name of the first mixture
    :param file2: the file name of the second mixture
    :param method: FOBI or FastICA
    :return: result of ICA
    """
    # input data
    samplingRate, signal1 = wavfile.read(file1)
    samplingRate, signal2 = wavfile.read(file2)

    # rescale signal
    signal1 = signal1 / 255.0 - 0.5  # uint8 takes values from 0 to 255
    signal2 = signal2 / 255.0 - 0.5  # uint8 takes values from 0 to 255

    a = signal1.shape
    n = a[0] * 1.0
    time = np.arange(0, n, 1)
    time = time / samplingRate
    time = time * 1000  # convert to milliseconds

    # draw the plot of mixture 1
    plt.figure()
    plt.plot(time, signal1, color='k')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (ms)')
    plt.title("Original signal 1")
    plt.savefig("output/Original signal 1.jpg")


    # draw the plot of mixture 2
    plt.figure()
    plt.plot(time, signal2, color='k')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (ms)')
    plt.title("Original signal 2")
    plt.savefig("output/Original signal 2.jpg")


    # x is our initial data matrix.
    mixture = [signal1, signal2]

    # Plot the signals from both sources to show correlations in the data.
    plt.figure()
    plt.plot(mixture[0], mixture[1], '*b')
    plt.ylabel('Signal 2')
    plt.xlabel('Signal 1')
    plt.title("Original data")
    plt.savefig("output/Original data.jpg")


    # pre-processing
    mixture_clean = preprocessing(mixture)

    source = ICA(mixture_clean, method)

    # draw the source
    plt.figure()
    plt.plot(time, source[0], color='k')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (ms)')
    plt.title("Generated signal 1")
    plt.savefig("output/Generated signal 1.jpg")


    plt.figure()
    plt.plot(time, source[1], color='k')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (ms)')
    plt.title("Generated signal 2")
    plt.savefig("output/Generated signal 2.jpg")

    wavfile.write('output1_fastica.wav', samplingRate, source[0])
    wavfile.write('output2_fastica.wav', samplingRate, source[1])
    pass


# analysis("dataset1/rsm2_mA.wav", "dataset1/rsm2_mB.wav", "FastICA")
# analysis("dataset2/rss_mA.wav", "dataset2/rss_mB.wav", "FOBI")
analysis("dataset3/rssd_A.wav", "dataset3/rssd_B.wav", "FastICA")
