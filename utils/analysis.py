import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve
from sklearn.preprocessing import label_binarize

def uncertainty_hist(df):

  target = df['label'].to_numpy().astype(np.int)
  pred = df['prediction'].to_numpy().astype(np.int)
  conf = df['confidence'].to_numpy().astype(np.float)
  kernel_distance_normal,kernel_distance_wrong= [],[]

  for i in range (len(target)):
    if target[i] == pred [i]:
      kernel_distance_normal.append(conf[i])
    else:
      kernel_distance_wrong.append(conf[i])
  return {"correct": np.array(kernel_distance_normal),"wrong": np.array(kernel_distance_wrong)}


def uncertainty_bar(n_classes, df):

  target = df['label'].to_numpy().astype(np.int)
  conf = df['confidence'].to_numpy().astype(np.float)
  sig = df['sigma'].to_numpy().astype(np.float)
  epistemic = np.empty((n_classes, 0)).tolist()
  aleatoric = np.empty((n_classes, 0)).tolist()

  for i in range (len(target)):
    epistemic[target[i]].append(conf[i])
    aleatoric[target[i]].append(sig[i])
  for i in range (n_classes):
    epistemic[i] = np.array(epistemic[i]).mean()
    aleatoric[i] = np.array(aleatoric[i]).mean()

  return {"epistemic":np.array(epistemic),"aleatoric":np.array(aleatoric)}


def conf_matrix(df):

  data = df.to_numpy()
  mat = confusion_matrix(data[:, 0], data[:, 1])
  m = len(mat)
  vis = np.zeros((m, m), dtype='O')

  for i in range(m):
    for j in range(m):
      vis[i, j] = [mat[i, j], []]
  
  for i, j, k, x, y  in data:
    if len(vis[int(i), int(j)][1]) < 16:
        vis[int(i), int(j)][1].append(y)

  return vis


def f1_per_class(df):

  data = df.to_numpy()
  return f1_score(data[:, 0], data[:, 1], average=None)


def f1_total(df):

  data = df.to_numpy()
  return f1_score(data[:, 0], data[:, 1], average='macro')

def roc(df1, logit):

  data = df1.to_numpy()
  y = label_binarize(data[:, 0], classes=list(np.unique(data[:, 0]))
  n_classes = y.shape[1]
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], logit[:, i])

  return fpr, tpr
