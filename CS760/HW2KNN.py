import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# D = np.loadtxt("data/D2z.txt")
# neigh = KNeighborsClassifier(n_neighbors=1)
# neigh.fit(D[:, :2], D[:, 2])
# # print(np.linspace(-2, 2, 41))
# # np.meshgrid
# # X = np.hstack((np.arange(-2, 2, .1).T, np.arange(-2, 2, .1).T))
# X = []
# for i in np.linspace(-2, 2, 41):
#     for j in np.linspace(-2, 2, 41):
#         X.append([i, j])
# X = np.asanyarray(X)
# y = neigh.predict(X)
# X0 = X[y == 0]
# X1 = X[y == 1]
# plt.plot(X0[:, 0], X0[:, 1], 'r.')
# plt.plot(X1[:, 0], X1[:, 1], 'b.')
# plt.plot(D[:, 0], D[:, 1], 'x')
# plt.show()

# D = np.genfromtxt("data/emails.csv", delimiter=',')[1:, 1:]
# for i in range(5):
#     train_ind = np.arange(5000)[1000*i:1000*(i+1)]
#     Dtest = D[1000*i:1000*(i+1)]
#     Dtrain = np.asanyarray([D[x] for x in np.arange(5000) if x not in train_ind])
#     neigh = KNeighborsClassifier(n_neighbors=1)
#     neigh.fit(Dtrain[:, :-1], Dtrain[:, -1])
#     y = neigh.predict(Dtest[:, :-1])
#     cor = y[y == Dtest[:, -1]]
#     print("Fold {0}: ".format(i))
#     print("Accuracy: {0}, ".format(len(cor)/len(y)), end="")
#     print("Precision: {0}, ".format(len(cor[cor == 1])/len(y[y == 1])), end="")
#     print("Recall: {0}".format(len(cor[cor == 1])/len(Dtest[:, -1][Dtest[:, -1] == 1])))

# D = np.genfromtxt("data/emails.csv", delimiter=',')[1:, 1:]
# for i in range(5):
#     train_ind = np.arange(5000)[1000*i:1000*(i+1)]
#     Dtest = D[1000*i:1000*(i+1)]
#     Dtrain = np.asanyarray([D[x] for x in np.arange(5000) if x not in train_ind])
#     clf = LogisticRegression(random_state=0).fit(Dtrain[:, :-1], Dtrain[:, -1])
#     # neigh.fit(Dtrain[:, :-1], Dtrain[:, -1])
#     y = clf.predict(Dtest[:, :-1])
#     cor = y[y == Dtest[:, -1]]
#     print("Fold {0}: ".format(i))
#     print("Accuracy: {0}, ".format(len(cor)/len(y)), end="")
#     print("Precision: {0}, ".format(len(cor[cor == 1])/len(y[y == 1])), end="")
#     print("Recall: {0}".format(len(cor[cor == 1])/len(Dtest[:, -1][Dtest[:, -1] == 1])))

# D = np.genfromtxt("data/emails.csv", delimiter=',')[1:, 1:]
# ks = [1,3,5,7,10]
# accs = []
# for k in ks:
#     avgs = np.asanyarray([0.0,0.0,0.0])
#     for i in range(5):
#         train_ind = np.arange(5000)[1000*i:1000*(i+1)]
#         Dtest = D[1000*i:1000*(i+1)]
#         Dtrain = np.asanyarray([D[x] for x in np.arange(5000) if x not in train_ind])
#         neigh = KNeighborsClassifier(n_neighbors=k)
#         neigh.fit(Dtrain[:, :-1], Dtrain[:, -1])
#         y = neigh.predict(Dtest[:, :-1])
#         cor = y[y == Dtest[:, -1]]
#         avgs += np.asanyarray([len(cor)/len(y), len(cor[cor == 1])/len(y[y == 1]), len(cor[cor == 1])/len(Dtest[:, -1][Dtest[:, -1] == 1])])/5
#     accs.append(avgs[0])
#     print(accs[-1])
# plt.plot(ks, accs)
# plt.show()

D = np.genfromtxt("data/emails.csv", delimiter=',')[1:, 1:]
i=0
train_ind = np.arange(5000)[1000*i:1000*(i+1)]
Dtest = D[1000*i:1000*(i+1)]
Dtrain = np.asanyarray([D[x] for x in np.arange(5000) if x not in train_ind])
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(Dtrain[:, :-1], Dtrain[:, -1])
pnn = neigh.predict_proba(Dtest[:, :-1])
clf = LogisticRegression(random_state=0).fit(Dtrain[:, :-1], Dtrain[:, -1])
p = clf.predict_proba(Dtest[:, :-1])
xs = []
ys = []
xsnn = []
ysnn = []
for t in np.linspace(-.01, 1.01, 102):
    ynn = np.asanyarray([(1 if z[0] >= t else 0) for z in pnn])
    y = np.asanyarray([(1 if z[0] >= t else 0) for z in p])
    cornn = ynn[ynn == Dtest[:, -1]]
    incnn = ynn[ynn != Dtest[:, -1]]
    cor = y[y == Dtest[:, -1]]  # correctly labeled predictions
    inc = y[y != Dtest[:, -1]]  # incorrectly labeled predictions
    tp_rate = len(cor[cor == 1])/len(Dtest[:, -1][Dtest[:, -1] == 1])
    fp_rate = len(inc[inc == 1])/len(Dtest[:, -1][Dtest[:, -1] == 0])
    xs.append(tp_rate)
    ys.append(fp_rate)
    tp_ratenn = len(cornn[cornn == 1])/len(Dtest[:, -1][Dtest[:, -1] == 1])
    fp_ratenn = len(incnn[incnn == 1])/len(Dtest[:, -1][Dtest[:, -1] == 0])
    xsnn.append(tp_ratenn)
    ysnn.append(fp_ratenn)
plt.plot(xs, ys, 'y')
plt.plot(xsnn, ysnn, 'b')
plt.show()
