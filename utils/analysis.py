import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_score, accuracy_score
from sklearn.preprocessing import label_binarize
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn
import utils.training as tr

img_size = 40
root_dir = os.path.dirname(os.path.realpath(__file__))
test_transform = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor()
])
def norm(a):
  a-=a.min()
  a/=a.max()
  return a
def mid(s):
  return [round((s[i]+s[i+1])/2,2) for i in range(0, len(s)-1, 1)]

#######################################################################################
## All functions have output as list and require tr.model

# Total dataset Scores
def uncertainty_hist(df):
  target = df['label'].to_numpy().astype(np.int).tolist()
  pred = df['prediction'].to_numpy().astype(np.int).tolist()
  conf = df['confidence'].to_numpy().astype(np.float).tolist()
  kernel_distance_normal,kernel_distance_wrong= [],[]

  for i in range (len(target)):
    if target[i] == pred [i]:
      kernel_distance_normal.append(conf[i])
    else:
      kernel_distance_wrong.append(conf[i])
  plt.ion()
  normal_,normal_l,_ = plt.hist(kernel_distance_normal,bins = 20,color=['#22FF22'])
  wrong_,wrong_l,_ = plt.hist(kernel_distance_wrong,bins = 20,color=['#22FF22'])
  plt.close('all')
  plt.ioff()
  return [normal_.tolist(),mid(normal_l.tolist()), wrong_.tolist(),mid(wrong_l.tolist())]


# per class scores
def uncertainty_bar(n_classes, df):
  target = df['label'].to_numpy().astype(np.int).tolist()
  conf = df['confidence'].to_numpy().astype(np.float).tolist()
  sig = df['sigma'].to_numpy().astype(np.float).tolist()
  epistemic = np.empty((n_classes, 0)).tolist()
  aleatoric = np.empty((n_classes, 0)).tolist()

  for i in range(len(target)):
    epistemic[target[i]].append(conf[i])
    aleatoric[target[i]].append(sig[i])
  for i in range(n_classes):
    if len(epistemic[i]):
        epistemic[i] = round(np.array(epistemic[i]).mean().item(),3)
    else:
        epistemic[i]=0
    if len(aleatoric[i]):
        aleatoric[i] = round(np.array(aleatoric[i]).mean().item(),3)
    else:
        epistemic[i]=0

  return [epistemic, np.arange(n_classes).tolist(), aleatoric, np.arange(n_classes).tolist()]


# per image score
def uncertainty_scores(path):
  usecuda = False
  with torch.no_grad():
    tr.model.eval()
    img = Image.open(path)
    img = test_transform(img)
    if (usecuda):
      img = img.cuda()
    img = img.unsqueeze(0)
    output = tr.model(img).squeeze().cpu()
    epistemic, aleatoric = round(output.max(0)[0].mean().item(),4), round(abs(tr.model.sigma.squeeze().item()),4)
    return epistemic, aleatoric


def conf_matrix(df):

  data = df.to_numpy()
  mat = confusion_matrix(data[:, 0], data[:, 1])
  m = len(mat)
  vis = np.zeros((m, m), dtype='O')
  #mat = np.random.rand(10,10)
  """for i in range(m):
    for j in range(m):
      vis[i, j] = [mat[i, j], []]


  for i, j, k, x, y  in data:
    if len(vis[int(i), int(j)][1]) < 16:
        vis[int(i), int(j)][1].append(y)"""

  cm = mat
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  cm = np.log(.0001 + cm)
  now = datetime.now()
  current = now.strftime("%H%M%S%f")

  root_dir = os.path.dirname(os.path.realpath(__file__))
  loc_path = os.path.join(root_dir,'..','data', 'analysis')
  plt.ion()
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title('Log of normalized Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  newimg = "confusion_"+str(current)+".png"
  img_name = os.path.join(loc_path,newimg )
  if os.path.exists(img_name):
      os.remove(img_name)
  plt.savefig(img_name)
  plt.close('all')
  plt.ioff()
  return os.path.join('data', 'analysis', newimg)


def f1_per_class(df):
  data = df.to_numpy()
  x = np.arange(len(np.unique(data[:,0]))).tolist()
  return [x, f1_score(data[:, 0], data[:, 1], average=None).tolist()]

def precision_per_class(df):
  data = df.to_numpy()
  x = np.arange(len(np.unique(data[:,0]))).tolist()
  return [x, precision_score(data[:,0],data[:,1],average=None).tolist()]

def acc_total(df):
  data = df.to_numpy()
  return accuracy_score(data[:, 0], data[:, 1]).tolist()

def f1_total(df):
  data = df.to_numpy()
  return f1_score(data[:, 0], data[:, 1], average='macro').tolist()

def roc(df1, logit):
  data = df1.to_numpy()
  y = label_binarize(data[:, 0], classes=list(np.unique(data[:, 0])))
  n_classes = y.shape[1]
  fpr = np.empty((n_classes, 0)).tolist()
  tpr = np.empty((n_classes, 0)).tolist()
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], logit[:, i])
    fpr[i] = fpr[i].tolist()
    tpr[i] = tpr[i].tolist()
  return fpr, tpr


def stn_view(path):
    usecuda = False
    with torch.no_grad():
      tr.model.eval()
      img = Image.open(path)
      img = test_transform(img)
      if (usecuda):
        img = img.cuda()
      img = img.unsqueeze(0)
      output = tr.model.stn(img).squeeze().cpu()

      now = datetime.now()
      current = now.strftime("%H%M%S%f")

      loc_path = os.path.join(root_dir,'..','data', 'analysis')
      newimg = "stn_"+str(current)+".png"
      img_name = os.path.join(loc_path, newimg)

      if os.path.exists(img_name):
        os.remove(img_name)
      save_image(output,img_name)
      return os.path.join('data', 'analysis',newimg)


def gradcam(path):
  usecuda = False
  tr.model.cam = True
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
      heatmap = norm(torch.relu(heatmap))
      map = F.interpolate(heatmap,(img.shape[2],img.shape[2]),mode='bilinear')
      map = map.squeeze().unsqueeze(2); img = img.squeeze().cpu().permute(1,2,0)*255; map = np.uint8(255 * map)
      map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
      superimpose = (img + st*map)
      superimpose /= superimpose.max()
      superimpose = superimpose.permute(2,0,1)

      now = datetime.now()
      current = now.strftime("%H%M%S%f")
      loc_path = os.path.join(root_dir,'..','data', 'analysis')
      newimg = "gradcam_"+str(current)+".png"
      img_name = os.path.join(loc_path, newimg)

      if os.path.exists(img_name):
        os.remove(img_name)
      save_image(superimpose,img_name)
  tr.model.cam = False
  return os.path.join('data', 'analysis',newimg)


def gradcam_noise(path):
  usecuda = False
  tr.model.cam = True
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
      heatmap = norm(torch.relu(heatmap))
      map = F.interpolate(heatmap,(img.shape[2],img.shape[2]),mode='bilinear')
      map = map.squeeze().unsqueeze(2); img = img.squeeze().cpu().permute(1,2,0)*255; map = np.uint8(255 * map)
      map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
      superimpose = (img + st*map)
      superimpose /= superimpose.max()
      superimpose = superimpose.permute(2,0,1)

      now = datetime.now()
      current = now.strftime("%H%M%S%f")
      loc_path = os.path.join(root_dir,'..','data', 'analysis')
      newimg = "gradcam_n_"+str(current)+".png"
      img_name = os.path.join(loc_path, newimg)
      if os.path.exists(img_name):
        os.remove(img_name)
      save_image(superimpose,img_name)
  tr.model.cam = False
  return os.path.join('data', 'analysis',newimg)

# Plots for weight viisualizations
def violinplot(hidden):
  i=0
  w = [];b = []
  tr.model.eval()
  plt.ion()
  for m in tr.model.modules():
      if isinstance(m, nn.Conv2d):
          w.append(m.weight.data.reshape(-1).cpu().detach().numpy()[0:864])
          b.append(m.bias.data.reshape(-1).cpu().detach().numpy()[0:32])
          i=i+1
          #print(m)
      if i==7+hidden:
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

  now = datetime.now()
  current = now.strftime("%H%M%S%f")
  loc_path = os.path.join(root_dir,'..','data', 'analysis')
  newimg = "violin_"+str(current)+".png"
  img_name = os.path.join(loc_path, newimg)
  if os.path.exists(img_name):
    os.remove(img_name)
  plt.savefig(img_name)
  plt.close('all')
  plt.ioff()
  return os.path.join('data', 'analysis',newimg)
