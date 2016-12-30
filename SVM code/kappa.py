__author__ = 'ravir'

import numpy as np
import pandas as pd

def score(obs, pred):

    op_df = pd.DataFrame(obs, columns=['obs'])
    op_df['pred'] = pred
    O_df = pd.DataFrame(op_df.groupby(['obs', 'pred']).size(), columns=['freq'])

    N=5
    O = np.zeros((N,N))
    W = np.array(O)
    for i in range(N):
        for j in range(N):
            O[i,j] = O_df['freq'].get((i,j), 0)
            W[i,j] = (i -j) ** 2
    W /= ((N - 1) ** 2)

    obs_v = op_df['obs'].value_counts()
    pred_v = op_df['pred'].value_counts()

    obs_l = []
    pred_l = []

    for i in range(N):
        obs_l.append( obs_v.get(i,0) )
        pred_l.append( pred_v.get(i,0) )

    obs_a = np.array(obs_l)
    pred_a = np.array(pred_l)

    E = np.outer(obs_a, pred_a)
    E = E * np.sum(W) / np.sum(E) # normalize

    k = 1

    num = 0
    den = 0
    for i in range(N):
        for j in range(N):
            num +=  W[i,j] * O[i,j]
            den += W[i,j] * E[i,j]

    kappa = 1.0 - (num * 1.0 / den)

    return kappa