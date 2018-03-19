import numpy as np
import h5py


def get_data():
    A01T = h5py.File("./project_datasets/A01T_slice.mat", 'r')
    X = np.copy(A01T["image"])
    y = np.copy(A01T["type"])
    y = y[0, 0:X.shape[0]:1]
    y = np.asarray(y, dtype=np.int32)
    X = X[:, :22, :]

    for i in range(len(y)):
        if y[i] == 769:
            y[i] = 0
        elif y[i] == 770:
            y[i] = 1
        elif y[i] == 771:
            y[i] = 2
        elif y[i] == 772:
            y[i] = 3
        else:
            y[i] = 0
    X = X.reshape(-1, 22*1000)
    y = y[~np.isnan(X).any(axis=1)]
    X = X[~np.isnan(X).any(axis=1)]
    X = X.reshape(-1, 22, 1000)

    # Define training and testing data
    t_size = np.array(range(X.shape[0]))
    np.random.shuffle(t_size)
    test_mask = t_size[:50]
    train_mask = t_size[50:]
    X_test = X[test_mask]
    y_test = y[test_mask]
    X_train = X[train_mask]
    y_train = y[train_mask]

    return X_test, y_test, X_train, y_train


def get_all_data():

    A01T = h5py.File("./project_datasets/A01T_slice.mat", 'r')
    X = np.copy(A01T["image"])
    y = np.copy(A01T["type"])
    y = y[0, 0:X.shape[0]:1]
    y = np.asarray(y, dtype=np.int32)
    X = X[:, :22, :]

    for i in range(len(y)):
        if y[i] == 769:
            y[i] = 0
        elif y[i] == 770:
            y[i] = 1
        elif y[i] == 771:
            y[i] = 2
        elif y[i] == 772:
            y[i] = 3
        else:
            y[i] = 0
    X = X.reshape(-1, 22*1000)
    y = y[~np.isnan(X).any(axis=1)]
    X = X[~np.isnan(X).any(axis=1)]
    X = X.reshape(-1, 22, 1000)

    # Define training and testing data
    t_size = np.array(range(X.shape[0]))
    np.random.shuffle(t_size)
    test_mask = t_size[:50]
    train_mask = t_size[50:]
    X_test = X[test_mask]
    y_test = y[test_mask]
    X_train = X[train_mask]
    y_train = y[train_mask]

    for j in range(2, 10):
        string_name = "./project_datasets/A0" + str(j) + "T_slice.mat"
        A01T = h5py.File(string_name, 'r')
        X = np.copy(A01T["image"])
        y = np.copy(A01T["type"])
        y = y[0, 0:X.shape[0]:1]
        y = np.asarray(y, dtype=np.int32)
        X = X[:, :22, :]

        for i in range(len(y)):
            if y[i] == 769:
                y[i] = 0
            elif y[i] == 770:
                y[i] = 1
            elif y[i] == 771:
                y[i] = 2
            elif y[i] == 772:
                y[i] = 3
            else:
                y[i] = 0
        X = X.reshape(-1, 22 * 1000)
        y = y[~np.isnan(X).any(axis=1)]
        X = X[~np.isnan(X).any(axis=1)]
        X = X.reshape(-1, 22, 1000)

        # Define training and testing data
        t_size = np.array(range(X.shape[0]))
        np.random.shuffle(t_size)
        test_mask = t_size[:50]
        train_mask = t_size[50:]
        X_test = np.concatenate((X_test, X[test_mask]), axis=0)
        y_test = np.concatenate((y_test, y[test_mask]), axis=0)
        X_train = np.concatenate((X_train, X[train_mask]), axis=0)
        y_train = np.concatenate((y_train, y[train_mask]), axis=0)

    return X_test, y_test, X_train, y_train


def crop_data(X_test, y_test, X_train, y_train):
    X_test_out = []
    y_test_out = []
    X_train_out = []
    y_train_out = []
    X_test_out = X_test[:, :, :500]
    y_test_out = y_test
    X_train_out = X_train[:, :, :500]
    y_train_out = y_train

    for i in range(1, 501, 50):
        X_test_out = np.concatenate((X_test_out, X_test[:, :, i:i+500]), axis=0)
        # y_test_out = np.concatenate((y_test_out, y_test))
        X_train_out = np.concatenate((X_train_out, X_train[:, :, i:i+500]), axis=0)
        y_train_out = np.concatenate((y_train_out, y_train))
    print(X_test.shape)
    return X_test_out, y_test_out, X_train_out, y_train_out
