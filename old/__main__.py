import os
import numpy as np
from model import Model
import tensorflow as tf

def hypergraph(np_labels):
    incidence = np_labels.T
    weightDiag = np.eye(np_labels.shape[0])
    edgesDiag = np.zeros((np_labels.shape[0], np_labels.shape[0]))
    vertexDiag = np.zeros((np_labels.shape[1], np_labels.shape[1]))
    for i in range(np_labels.shape[1]):
        vertexDiag[i, i] = np.sum(np_labels[:, i])
    for i in range(np_labels.shape[0]):
        edgesDiag[i, i] = np.sum(np_labels[i, :])
    return [incidence, weightDiag, edgesDiag, vertexDiag]

def main():
    model = Model()