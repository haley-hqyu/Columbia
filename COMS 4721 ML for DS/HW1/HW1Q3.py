import pandas as pd
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from math import sqrt


# read file
X_train = pd.read_csv("X_train.csv", header = None).values
X_test = pd.read_csv("X_test.csv", header = None).values
y_train = pd.read_csv("y_train.csv", header = None).values
y_test = pd.read_csv("y_test.csv", header = None).values


# standardize X and y, but here X is fine, so we only standarize y
y_train = y_train - np.mean(y_train)
y_test = y_test - np.mean(y_train)


# question (a)
U, S, Vt = la.svd(X_train, full_matrices=False)
V = np.transpose(Vt)
result_a = list()
for para in range(5001):
    # calculate df(para)
    df_para = sum([temp * temp / (para + temp * temp) for temp in S])
    temp_list = [df_para]
    # calculate parameter
    S_para = np.diag([temp / (para + temp * temp) for temp in S])
    wrr = np.transpose(V.dot(S_para).dot(np.transpose(U)).dot(y_train))[0]
    temp_list.extend(list(wrr))
    result_a.append(temp_list)

# plot wrr wrt df(para)
result_a_df = pd.DataFrame(result_a, columns = ["df", "cylinders", "displacement", "horsepower",
                                                "weight", "acceleration", "year made", "constant"])
result_a_df.plot(x="df", y=["cylinders", "displacement", "horsepower", "weight", "acceleration",
                            "year made", "constant"], kind="line")
plt.xlabel('df(lambda)')
plt.ylabel('parameter')
plt.show()


# question (c)
result_c = list()
for para in range(51):
    # calculate parameter
    S_para = np.diag([temp / (para + temp * temp) for temp in S])
    wrr = np.transpose(V.dot(S_para).dot(np.transpose(U)).dot(y_train))[0]
    # calculate predict
    y_predict = np.dot(X_test, wrr)
    diff = y_predict - np.transpose(y_test)[0]
    rmse = sqrt(sum([temp * temp for temp in diff])/42.0)
    result_c.append(rmse)

# plot
plt.plot(range(51), result_c)
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.show()

# question(d)
X_train_feature = X_train[:, : -1]
X_test_feature = X_test[:, : -1]


def cal_rmse(p, para):
    if p == 1:
        X_train_p = X_train
        X_test_p = X_test
    else:
        X_train_p = X_train_feature
        train_temp = X_train_feature
        X_test_p = X_test_feature
        test_temp = X_test_feature
        for order in range(1, p):
            # generate data
            train_temp = np.multiply(train_temp, X_train_feature)
            test_temp = np.multiply(test_temp, X_test_feature)
            # standardize
            X_train_append_mean = [np.transpose(train_temp)[ind].mean() for ind in range(len(train_temp[0]))]
            X_train_append_var = [np.std(np.transpose(train_temp)[ind]) for ind in range(len(train_temp[0]))]
            X_train_append = np.divide(np.subtract(train_temp, np.array([X_train_append_mean for temp in
                                                                         range(len(X_train))])),
                                       np.array([X_train_append_var for temp in range(len(X_train))]))
            X_test_append = np.divide(np.subtract(test_temp, np.array([X_train_append_mean for temp in
                                                                       range(len(X_test))])),
                                      np.array([X_train_append_var for temp in range(len(X_test))]))
            # concate
            X_train_p = np.concatenate((X_train_p, X_train_append), axis=1)
            X_test_p = np.concatenate((X_test_p, X_test_append), axis=1)

        # constant
        X_train_p = np.concatenate((X_train_p, np.full((len(X_train_p), 1), 1)), axis=1)
        X_test_p = np.concatenate((X_test_p, np.full((len(X_test_p), 1), 1)), axis=1)

    # RR
    # calculate parameter
    U, S, Vt = la.svd(X_train_p, full_matrices=False)
    V = np.transpose(Vt)
    S_para = np.diag([temp / (para + temp * temp) for temp in S])
    wrr = np.transpose(V.dot(S_para).dot(np.transpose(U)).dot(y_train))[0]
    # calculate predict
    y_predict = np.dot(X_test_p, wrr)
    diff = y_predict - np.transpose(y_test)[0]
    rmse = sqrt(sum([temp * temp for temp in diff]) / 42.0)
    return rmse

# plot
result_d1 = list()
result_d2 = list()
result_d3 = list()
for para in range(101):
    result_d1.append(cal_rmse(1, para))
    result_d2.append(cal_rmse(2, para))
    result_d3.append(cal_rmse(3, para))

plt.plot(range(101), result_d1, color = "blue", label = "p=1")
plt.plot(range(101), result_d2, color = "red", label = "p=2")
plt.plot(range(101), result_d3, color = "green", label = "p=3")
plt.legend()
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.show()
