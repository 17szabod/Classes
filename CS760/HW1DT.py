import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import itertools
from sklearn.tree import DecisionTreeClassifier


class Tree():
    def __init__(self, feature, split):
        self.parent = None
        self.feature = feature
        self.split = split
        self.children = []
        self.leaf = type(split) == str

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def set_split(self, s):
        self.split = s

    def set_feature(self, feat):
        self.feature = feat

    def __str__(self):
        return self.str_help(0)

    def str_help(self, depth):
        if self.leaf or type(self.split) is str:
            return "y=" + self.split + "\\\\"
        to_ret = ""
        to_ret += "If x_{0} \leq {1}:\\\\\n".format(self.feature+1, self.split)
        to_ret += (depth+1)*"\t" + self.children[0].str_help(depth+1) + "\n" + "\t"*depth + "Else:\\\\\n" + "\t"*(depth+1) + self.children[1].str_help(depth+1)
        return to_ret


def FindBestSplit(D, C1, C2):
    max_gain_ratio1 = 0
    max_gain_ratio2 = 0
    best_split1 = None
    best_split2 = None
    for S in C1:
        x = GainRatio(D, S, 0)
        if x > max_gain_ratio1:
            max_gain_ratio1 = x
            best_split1 = S
    for S in C2:
        x = GainRatio(D, S, 1)
        if x > max_gain_ratio2:
            max_gain_ratio2 = x
            best_split2 = S
    if max_gain_ratio2 == max_gain_ratio1 == 0:
        return None, None
    return (best_split1, 0) if best_split1 is not None and (best_split2 is None or best_split1 >= best_split2) else (best_split2, 1)


def DetCandSplits(D):
    first_splits = []
    second_splits = []
    D1 = sorted(D, key=lambda x: x[0])
    cur_val = D1[0][2]
    for i in range(1, len(D1)):
        if D1[i][2] != cur_val:
            cur_val = D1[i][2]
            first_splits.append((D1[i][0] + D1[i - 1][0]) / 2)
    D2 = sorted(D, key=lambda x: x[1])
    cur_val = D2[0][2]
    for i in range(1, len(D2)):
        if D2[i][2] != cur_val:
            cur_val = D2[i][2]
            second_splits.append((D2[i][1] + D2[i - 1][1]) / 2)
    return first_splits, second_splits


def GainRatio(D, S, feat):
    H_Y = entropy([1 - np.count_nonzero(D[:, 2]) / len(D), np.count_nonzero(D[:, 2]) / len(D)], base=2)
    if len(D[D[:, feat] <= S]) == 0 or len(D[D[:, feat] > S]) == 0:
        # Split is useless, skip
        # print("This shouldn't happen, right?")
        return 0
    H_YX = len(D[D[:, feat] <= S]) / len(D) * entropy(
        [1 - np.count_nonzero(D[D[:, feat] <= S][:, 2]) / len(D[D[:, feat] <= S]),
         np.count_nonzero(D[D[:, feat] <= S][:, 2]) / len(D[D[:, feat] <= S])], base=2) \
           + len(D[D[:, feat] > S]) / len(D) * entropy(
        [1 - np.count_nonzero(D[D[:, feat] > S][:, 2]) / len(D[D[:, feat] > S]),
         np.count_nonzero(D[D[:, feat] > S][:, 2]) / len(D[D[:, feat] > S])], base=2)
    gain_ratio = (H_Y - H_YX) / entropy([len(D[D[:, feat] <= S]) / len(D), len(D[D[:, feat] > S]) / len(D)], base=2)
    # print("Cut {0} has gain ratio {1}".format(S, gain_ratio)) if len(D) == 11 else ""
    return gain_ratio


def MakeSubTree(D, root):
    C1, C2 = DetCandSplits(D)
    if len(C1) == 0 or len(C2) == 0:  # This split determines everything, zero entropy split
        root = Tree(0 if len(C1) == 0 else 1, str(D[0][2]))
        return root
    S, feat = FindBestSplit(D, C1, C2)
    if S is None:
        root = Tree(0, str(D[0][2]))
        return root
    root.set_split(S)
    root.set_feature(feat)
    new_rt1 = Tree(-1, -1)
    new_rt2 = Tree(-1, -1)
    root.add_child(MakeSubTree(D[D[:, feat] <= S], new_rt1))
    root.add_child(MakeSubTree(D[D[:, feat] > S], new_rt2))
    return root


def decide(tree, x):
    if tree.leaf or type(tree.split) is str:
        return tree.split
    if x[tree.feature] <= tree.split:
        return decide(tree.children[0], x)
    else:
        return decide(tree.children[1], x)


def count(tree):
    if tree.leaf or type(tree.split) is str:
        return 1
    return count(tree.children[0]) + count(tree.children[1])


if __name__ == '__main__':
    x = np.random.sample(100)*20
    y = np.sin(x)

    def f(x_i):
        return sum([y[i] * np.prod([(x_i - x[j])/(x[i] - x[j]) for j in range(len(x)) if j != i]) for i in range(len(x))])


    x2 = np.random.sample(100) * 20
    y2 = np.sin(x)

    print("Train error (MSE): " + str(np.linalg.norm(np.asanyarray([f(x[i]) for i in range(len(y))]) - y)))
    print("Test error (MSE): " + str(np.linalg.norm(np.asanyarray([f(x2[i]) for i in range(len(y))]) - y2)))

    for e in [0, 1e-30, 1e-8, 1e-6, 1e-2]:
        eps = np.random.multivariate_normal([0, ]*100, e*np.eye(len(y)))
        x3 = x+eps
        y3 = np.sin(x3)
        print("Error for epsilon\\tilde \mathcal N(0,{0})^100: {1}".format(e, np.linalg.norm(np.asanyarray([f(x3[i]) for i in range(len(y))]) - y3)))

    exit(0)
    D = np.loadtxt("data/Dbig.txt")
    perm = np.random.permutation(len(D))
    training_D = D[perm[:8192]]
    test_D = D[perm[8192:]]
    D_32 = D[perm[:32]]
    D_128 = D[perm[:128]]
    D_512 = D[perm[:512]]
    D_2048 = D[perm[:2048]]
    xs = []
    ys = []
    for D_prime in [D_32, D_128, D_512, D_2048, training_D]:
        # reds = []
        # blues = []
        # tree = Tree(-1, -1)
        # tree = MakeSubTree(D_prime, tree)
        clf = DecisionTreeClassifier()
        clf.fit(D_prime[:, :2], D_prime[:, 2])
        acc = clf.score(test_D[:, :2], test_D[:, 2])
        # good_ct = 0
        # for test_pt in test_D:
        #     ans = decide(tree, test_pt[:2])
        #     if ans == str(test_pt[2]):
        #         good_ct += 1
            # reds.append(test_pt[:2]) if ans == '1.0' else blues.append(test_pt[:2])
        xs.append(len(D_prime))
        ys.append(1-acc)
        # reds = np.asanyarray(reds)
        # blues = np.asanyarray(blues)
        # plt.plot(reds[:, 0], reds[:, 1], 'ro')
        # plt.plot(blues[:, 0], blues[:, 1], 'bo')
        # plt.title('Decisions on test set for n=' + str(len(D_prime)))
        # plt.show()
        print("n={0}, number of nodes={1}, err_n={2}".format(len(D_prime), clf.get_n_leaves(), 1-acc))
    plt.plot(xs, ys, 'ro-')

    # print(tree)
    plt.show()
