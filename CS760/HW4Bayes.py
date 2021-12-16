import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer

chars = 'a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p.q.r.s.t.u.v.w.x.y.z. '.split('.')
thetas = {}
for c in ['j', 's', 'e']:
    vectorizer = CountVectorizer(analyzer='char', vocabulary=chars)
    total_counts = np.zeros(len(chars))
    total_lens = np.zeros(len(chars))
    for i in range(10):
        with open('data/languageID/' + c + str(i) + '.txt') as infile:
            trim_bod = infile.read().replace('\n', '')
            X = vectorizer.fit_transform([trim_bod])
            total_counts += X
            total_lens += np.sum(X)
    theta = (total_counts + 1/2) / (total_lens + 27/2)
    thetas[c] = theta
    print("Theta_{0} = {1}".format(c, theta))

vectorizer = CountVectorizer(analyzer='char', vocabulary=chars)
with open('data/languageID/e10.txt') as infile:
    trim_bod = infile.read().replace('\n', '')
    X = vectorizer.fit_transform([trim_bod]).toarray()
    print(X)
    # theta = (X + 1 / 2) / (tot_len + 27 / 2)
    posteriors = {}
    for c in ['e', 'j', 's']:
        p_hat = np.sum(np.dot(X.T, np.log(thetas[c])))
        print("log(p(x|y={0})) = {1}".format(c, p_hat))
        p_hat_y = p_hat - 3  # multiply by prior P(y=c)=1/3
        posteriors[c] = p_hat_y
    log_sum = np.logaddexp(posteriors['e'], posteriors['j'])
    log_sum = np.logaddexp(log_sum, posteriors['s'])
    for c in ['e', 'j', 's']:
        print("log(p(y={0}|x)) = {1}".format(c, np.exp(posteriors[c]-log_sum)))
# Test:
confusion = np.zeros((3, 3))
for c in ['j', 's', 'e']:
    vectorizer = CountVectorizer(analyzer='char', vocabulary=chars)
    total_counts = np.zeros(len(chars))
    total_lens = np.zeros(len(chars))
    for i in range(10):
        with open('data/languageID/' + c + str(i+10) + '.txt') as infile:
            trim_bod = infile.read().replace('\n', '')
            X = vectorizer.fit_transform([trim_bod]).toarray()
            posteriors = {}
            for c1 in ['e', 'j', 's']:
                p_hat = np.sum(np.dot(X.T, np.log(thetas[c1])))
                # print("log(p(x|y={0})) = {1}".format(c1, p_hat))
                p_hat_y = p_hat - 3  # multiply by prior P(y=c)=1/3
                posteriors[c1] = p_hat_y
            log_sum = np.logaddexp(posteriors['e'], posteriors['j'])
            log_sum = np.logaddexp(log_sum, posteriors['s'])
            for c1 in ['j', 's', 'e']:
                # print("log(p(y={0}|x)) = {1}".format(c, np.exp(posteriors[c] - log_sum)))
                if np.exp(posteriors[c1] - log_sum) >= 1/3:
                    confusion[['e', 'j', 's'].index(c1), ['e', 'j', 's'].index(c)] += 1
print(confusion)

vectorizer = CountVectorizer(analyzer='char', vocabulary=chars)
with open('data/languageID/e10.txt') as infile:
    trim_bod = infile.read().mreplace('\n', '')
    trim_bod = np.random.permutation(list(trim_bod))
    trim_bod = ''.join(trim_bod)
    print(trim_bod)
    X = vectorizer.fit_transform([trim_bod]).toarray()
    print(X)
    # theta = (X + 1 / 2) / (tot_len + 27 / 2)
    posteriors = {}
    for c in ['e', 'j', 's']:
        p_hat = np.sum(np.dot(X.T, np.log(thetas[c])))
        print("log(p(x|y={0})) = {1}".format(c, p_hat))
        p_hat_y = p_hat - 3  # multiply by prior P(y=c)=1/3
        posteriors[c] = p_hat_y
    log_sum = np.logaddexp(posteriors['e'], posteriors['j'])
    log_sum = np.logaddexp(log_sum, posteriors['s'])
    for c in ['e', 'j', 's']:
        print("log(p(y={0}|x)) = {1}".format(c, np.exp(posteriors[c]-log_sum)))
