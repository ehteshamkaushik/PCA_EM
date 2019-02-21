import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal as mvn

filename = "data.txt"
data = []
with open(filename) as fp:
    for line in fp:
        temp = line.split()
        tempList = [float(temp[i]) for i in range(len(temp))]
        data.append(tempList)


data = np.asarray(data)
print(data.shape)
cov = np.cov(data.transpose())
print(cov.shape)

eigval, eigvect = np.linalg.eig(cov)
eigvect = eigvect.transpose()

print(eigvect.shape)
pca = []
pca_count = 2
for i in range(len(data)):
    temp = [np.dot(eigvect[0], data[i].T), np.dot(eigvect[1], data[i].T)]
    pca.append(temp)
pca = np.asarray(pca)

N = len(pca)
K = 3
w = np.random.random(K)
w = w/w.sum()


cov_mat = []
for i in range(K):
    #temp1 = [0.0 for j in range(pca_count)]
    #mean.append(temp1)
    x=np.random.random((2,2))
    cov_mat.append(np.eye(2))

mean = np.random.random((3, 2))
all_prob = np.zeros((N, K))

'''
def calc_prob():
    for j in range(N):
        prob = []
        for i in range(K):
            temp_mean = mean[i]
            ep = cov_mat[i]
            ep_inv = np.linalg.inv(ep)
            diff = np.subtract(pca[j], temp_mean)
            diff_trans = np.asarray(diff).transpose()
            val = np.matmul(diff_trans, np.matmul(ep_inv, diff))
            val = val*(-0.5)
            val = math.exp(val)
            val = val/(math.sqrt(math.pow(2*3.1416, pca_count) * np.linalg.det(ep)))
            prob.append(val)
        all_prob[j] = prob.copy()
    #print(all_prob)
'''


def calc_prob():
    for i in range(N):
        temp = []
        for j in range(K):
            temp.append(mvn.pdf(pca[i], mean[j], cov_mat[j]))
        all_prob[i] = temp


cd_latent_factor = []


def e_step():
    cd_latent_factor.clear()
    for k in range(N):
        p_i_k = []
        sum = 0
        for j in range(K):
            sum += (w[j] * all_prob[k][j])
        #print(sum)
        for i in range(K):
            p_i_k.append(w[i] * all_prob[k][i]/sum)
        #print(p_i_k)
        cd_latent_factor.append(p_i_k)
    #print(cd_latent_factor)


def m_step():
    e_step()
    #print("Latent factor : ", cd_latent_factor)
    for i in range(K):
        temp_mean = [0 for j in range(pca_count)]
        sum = 0
        for j in range(N):
            x = pca[j]
            temp_mean = np.add(temp_mean, cd_latent_factor[j][i] * np.asarray(x))
            sum += cd_latent_factor[j][i]
        mean[i] = temp_mean/sum

    for i in range(K):
        temp_var = np.zeros((pca_count, pca_count))
        #print(temp_var)
        #temp_var = np.identity(pca_count)
        sum = 0
        for j in range(N):
            x = pca[j]
           # print("X : ", x)
            temp_mean = mean[i]
            #print("Mean : ", mean[i])
            diff = np.subtract(x, temp_mean).reshape(2, 1)
            #print("Diff : ", diff.shape)
            diff_trans = np.asarray(diff).transpose()
            #print("Diff Trans : ", diff_trans.shape)
            temp = np.matmul(diff, diff_trans)
            #print("Temp : ", temp)
            temp = cd_latent_factor[j][i] * temp
            temp_var = np.add(temp_var, temp)
            sum += cd_latent_factor[j][i]
        #print("Temp : ", temp_var)
        cov_mat[i] = temp_var/sum

    for i in range(K):
        sum = 0
        for j in range(N):
            sum += cd_latent_factor[j][i]
        w[i] = sum/N


def calc_log_likelihood():
    log_likelihood = 0
    calc_prob()
    print(all_prob)
    for i in range(N):
        sum_in = 0
        for j in range(K):
            sum_in += (w[j]*all_prob[i][j])
        log_val = np.log(sum_in)
        log_likelihood += log_val
    #print("Here")
    return log_likelihood


l = calc_log_likelihood()


def plot():
    index = []
    for i in range(N):
        max = cd_latent_factor[i][0]
        idx = 0
        for j in range(1, K):
            if max < cd_latent_factor[i][j]:
                max = cd_latent_factor[i][j]
                idx = j
        index.append(idx)

    for i in range(N):
        if (index[i]) == 0:
            plt.plot(pca[i][0], pca[i][1], 'ro')
        elif (index[i]) == 1:
            plt.plot(pca[i][0], pca[i][1], 'go')
        elif (index[i]) == 2:
            plt.plot(pca[i][0], pca[i][1], 'bo')

    plt.show()


count = 0
while True:
    #x = input()
    print("Count : ", count)
    count += 1
    m_step()
    #print("Mean : ", mean)
    #print("Cov : ", cov_mat)
    l_new = calc_log_likelihood()
    print(l_new)
    #print("diff : ", abs(l_new - l))

    if abs(l_new - l) == 0:
        break
    l = l_new

    #x = input()


plot()

