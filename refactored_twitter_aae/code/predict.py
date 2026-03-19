import numpy as np
import os

vocabfile = "../model/model_vocab.txt"
modelfile = "../model/model_count_table.txt"

K = 0
wordprobs = None
w2num = None

def load_model():
    """Idempotent"""
    global vocab, w2num, N_wk, N_k, wordprobs, N_w, K, modelfile, vocabfile
    if wordprobs is not None:
        # assume already loaded
        return

    N_wk = np.loadtxt(modelfile)
    N_w = N_wk.sum(1)
    N_k = N_wk.sum(0)
    K = len(N_k)
    wordprobs = (N_wk + 1) / N_k

    with open(vocabfile, encoding="utf-8") as f:
        vocab = [L.split("\t")[-1].strip() for L in f]
    w2num = {w: i for i, w in enumerate(vocab)}
    assert len(vocab) == N_wk.shape[0]

def infer_cvb0(invocab_tokens, alpha, numpasses):
    global K, wordprobs, w2num
    doclen = len(invocab_tokens)

    # initialize with likelihoods
    z = np.zeros((doclen, K))
    for i, token in enumerate(invocab_tokens):
        if token in w2num:
            z[i] = wordprobs[w2num[token]]
        else:
            z[i] = 1.0 / K
    z /= z.sum(1)[:, None]

    # iterate
    for _ in range(numpasses):
        for i, token in enumerate(invocab_tokens):
            if token in w2num:
                z[i] = wordprobs[w2num[token]] * (alpha + z.sum(0))
            else:
                z[i] = alpha + z.sum(0)
            z[i] /= z[i].sum()

    return z.sum(0) / z.sum()

def predict_language(tokens, alpha=0.01, numpasses=5):
    global w2num
    invocab_tokens = [t for t in tokens if t in w2num]
    if len(invocab_tokens) == 0:
        return None
    return infer_cvb0(invocab_tokens, alpha, numpasses)