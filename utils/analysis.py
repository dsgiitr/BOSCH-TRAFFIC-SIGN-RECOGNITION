import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_score, accuracy_score
from sklearn.preprocessing import label_binarize
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn
import utils.training as tr

sz = 40
test_transform = transforms.Compose([
    transforms.Resize((sz,sz)),
    transforms.ToTensor()
])
def norm(a):
  a-=a.min()
  a/=a.max()
  return a



# Total dataset Scores
def uncertainty_hist(df):

  target = df['label'].to_numpy().astype(np.int)
  pred = df['prediction'].to_numpy().astype(np.int)
  conf = df['confidence'].to_numpy().astype(np.float)
  kernel_distance_normal,kernel_distance_wrong= [],[]
  normal_labels, wrong_labels = [], []

  for i in range(len(target)):
    if target[i] == pred[i]:
      normal_labels.append(target[i])
      kernel_distance_normal.append(conf[i])
    else:
      wrong_labels.append(target[i])
      kernel_distance_wrong.append(conf[i])
  return [kernel_distance_normal, normal_labels, kernel_distance_normal, wrong_labels]


# per class scores
def uncertainty_bar(n_classes, df):

  target = df['label'].to_numpy().astype(np.int)
  conf = df['confidence'].to_numpy().astype(np.float)
  sig = df['sigma'].to_numpy().astype(np.float)
  epistemic = np.empty((n_classes, 0)).tolist()
  aleatoric = np.empty((n_classes, 0)).tolist()
  epistemic_label, aleatoric_label = [], []

  for i in range(len(target)):
    epistemic_label.append(target[i])
    aleatoric_label.append(target[i])
    epistemic[target[i]].append(conf[i])
    aleatoric[target[i]].append(sig[i])
  for i in range(n_classes):
    epistemic[i] = np.array(epistemic[i]).mean()
    aleatoric[i] = np.array(aleatoric[i]).mean()

  return [epistemic, epistemic_label, aleatoric, aleatoric_label]


# per image score
def uncertainty_scores(path, usecuda = True):
  with torch.no_grad():
    tr.model.eval()
    img = Image.open(path)
    img = test_transform(img)
    if (usecuda):
      img = img.cuda()
    img = img.unsqueeze(0)
    output = tr.model(img).squeeze().cpu()
    epistemic, aleatoric = output.max(0)[0].mean().item(), abs(tr.model.sigma.squeeze().item())
    return epistemic, aleatoric


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

  cm = mat
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  cm = np.log(.0001 + cm)

  loc_path = os.path.join('data', 'analysis')
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title('Log of normalized Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.ioff()
  img_name = os.path.join(loc_path, "confusion.png")
  if os.path.exists(img_name):
      os.remove(img_name)
  plt.savefig(img_name)
  return img_name


def f1_per_class(df):

  data = df.to_numpy()
  return [data[:, 0], f1_score(data[:, 0], data[:, 1], average=None)]

def precision_per_class(df):
  data = df.to_numpy()
  return [data[:, 0], precision_score(data[:,0],data[:,1],average=None)]

def acc_total(df):
  data = df.to_numpy()
  return accuracy_score(data[:, 0], data[:, 1])

def f1_total(df):
  data = df.to_numpy()
  return f1_score(data[:, 0], data[:, 1], average='macro')

def roc(df1, logit):
  data = df1.to_numpy()
  y = label_binarize(data[:, 0], classes=list(np.unique(data[:, 0])))
  n_classes = y.shape[1]
  fpr = np.empty((n_classes, 0)).tolist()
  tpr = np.empty((n_classes, 0)).tolist()
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], logit[:, i])
  return fpr, tpr


def stn_view(path, usecuda = True):
    with torch.no_grad():
      tr.model.eval()
      img = Image.open(path)
      img = test_transform(img)
      if (usecuda):
        img = img.cuda()
      img = img.unsqueeze(0)
      output = tr.model.stn(img).squeeze().cpu()
      loc_path = os.path.join('data', 'analysis')
      img_name = os.path.join(loc_path, "stn.png")
      if os.path.exists(img_name):
        os.remove(img_name)
      save_image(output,img_name)
      return img_name


def gradcam(path, usecuda = True):
  model.cam = True
  st = 0.7
  tr.model.eval()
  img = Image.open(path)
  img = test_transform(img).unsqueeze(0)
  if usecuda:
    img = img.cuda()
  out = tr.model(img)
  pred = out.argmax(1).squeeze()

  out[:,pred].backward()
  gradients = tr.model.grad
  pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
  activations = tr.model.activations(img).detach()
  for i in range(activations.shape[1]):
    activations[:, i, :, :] *= pooled_gradients[i]

  loc_path = os.path.join('data', 'analysis')

  with torch.no_grad():
      heatmap = torch.mean(activations, dim=1).cpu().unsqueeze(1)
      heatmap = torch.norm(torch.relu(heatmap))
      map = F.interpolate(heatmap,(img.shape[2],img.shape[2]),mode='bilinear')
      map = map.squeeze().unsqueeze(2); img = img.squeeze().cpu().permute(1,2,0)*255; map = np.uint8(255 * map)
      map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
      superimpose = (img + st*map)
      superimpose /= superimpose.max()
      superimpose = superimpose.permute(2,0,1)
      img_name = os.path.join(loc_path, "gradcam.png")
      if os.path.exists(img_name):
        os.remove(img_name)
      save_image(superimpose,img_name)
  model.cam = False
  return img_name


def gradcam_noise(path, usecuda = True):
  model.cam = True
  st = 0.7
  tr.model.eval()
  img = Image.open(path)
  img = test_transform(img).unsqueeze(0)
  if usecuda:
    img = img.cuda()
  out = tr.model(img)
  pred = out.argmax(1).squeeze()

  tr.model.sigma.backward()
  gradients = tr.model.grad
  pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
  activations = tr.model.activations(img).detach()
  for i in range(activations.shape[1]):
    activations[:, i, :, :] *= pooled_gradients[i]

  loc_path = os.path.join('data', 'analysis')

  with torch.no_grad():
      heatmap = torch.mean(activations, dim=1).cpu().unsqueeze(1)
      heatmap = torch.norm(torch.relu(heatmap))
      map = F.interpolate(heatmap,(img.shape[2],img.shape[2]),mode='bilinear')
      map = map.squeeze().unsqueeze(2); img = img.squeeze().cpu().permute(1,2,0)*255; map = np.uint8(255 * map)
      map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
      superimpose = (img + st*map)
      superimpose /= superimpose.max()
      superimpose = superimpose.permute(2,0,1)
      img_name = os.path.join(loc_path, "gradcam_n.png")
      if os.path.exists(img_name):
        os.remove(img_name)
      save_image(superimpose,img_name)
  model.cam = False
  return img_name


def violinplot():
  i=0
  w = [];b = []
  tr.model.eval()
  for m in tr.model.modules():
      if isinstance(m, nn.Conv2d):
          w.append(m.weight.data.reshape(-1).cpu().detach().numpy()[0:864])
          b.append(m.bias.data.reshape(-1).cpu().detach().numpy()[0:32])
          i=i+1
          #print(m)
      if i==7:
          break
  def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
        return lower_adjacent_value, upper_adjacent_value
  def violinplots(pl,title):
    plt.title(title)
    parts = plt.violinplot(
            pl, showmeans=False, showmedians=False,
            showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    quartile1, medians, quartile3 = np.percentile(pl, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(pl, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    inds = np.arange(1, len(medians) + 1)
    plt.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
  plt.figure(figsize=(10,9))
  plt.subplot(2,1,1)
  violinplots(w,'Violin plot of Conv Weights')
  plt.subplot(2,1,2)
  violinplots(b,'Violin plot of Conv Biases')
  loc_path = os.path.join('data', 'analysis')
  img_name = os.path.join(loc_path, "violinplot.png")
  if os.path.exists(img_name):
    os.remove(img_name)
  plt.savefig(img_name)
  return img_name
