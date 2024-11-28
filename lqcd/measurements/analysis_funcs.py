import numpy as np
import matplotlib.pyplot as plt



def jackknife_relist(thing):
    N = thing.shape[0] - 1
    mean = thing[0]
    sigma = np.sqrt(np.sum((thing[1:] - mean) ** 2, axis=0) * (N - 1) / N)
    return sigma


def bootstrap_relist(thing):
    tilde = np.mean(thing[1:], axis=0)
    sigma = np.sqrt(np.mean((thing[1:] - tilde) ** 2, axis=0))
    return sigma


def bootstrap_cov(thing):
    N = thing.shape[0] - 1
    return np.cov(np.transpose(thing[1:])) * (N-1) / N


def resamplelist(length,tparams):
    if (tparams['tech'] == 'bootstrap'):
        np.random.seed(tparams['seed'])
        return np.concatenate((np.reshape(np.arange(length), (1,-1)), np.random.randint(0,length,(tparams['Nbs'],tparams['Mbs']))))
    elif (tparams['tech'] == 'jackknife'):
        jk_lst = np.zeros([length+1,length],dtype=int)
        jk_lst[0] = np.arange(length)
        for i in range(length):
            jk_lst[i+1] = np.concatenate((np.arange(0,i),np.arange(i+1,length),[10000]))
        return jk_lst


def cal_mean(thing):
    return thing[0]


def cal_err(thing,tech):
    if (tech == 'bootstrap'):
        return bootstrap_relist(thing)
    elif (tech == 'jackknife'):
        return jackknife_relist(thing)


def cal_mass(data,mtype='exp',tau=1):
    data = data.real
    T = data.shape[0]
    if (mtype == 'exp') or (mtype == 'Gexp'):
        meff = 1./tau * np.log(data / np.roll(data,-tau))
    elif (mtype == 'cosh') or (mtype == 'sinh') or (mtype == 'Gcosh') or (mtype == 'Gsinh'):
        meff = 1./tau * np.arccosh((np.roll(data,-tau)+np.roll(data,tau))/2/data)
    return meff
