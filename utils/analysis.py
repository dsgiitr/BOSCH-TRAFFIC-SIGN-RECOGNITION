import pandas as pd
import numpy as np
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix, f1_score, roc_curve
from sklearn.preprocessing import label_binarize
from torchvision import transforms
test_transform = transforms.Compose([
    transforms.Resize((sz,sz)),
    transforms.ToTensor()
])


# Total dataset Scores
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

# per class scores
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

# per image score
def uncertainty_scores(path, model, usecuda = True):
  with torch.no_grad():
    model.eval()
    img = Image.open(path)
    img = test_transform(img)
    if (usecuda):
      img = img.cuda()
    img = img.unsqueeze(0)
    output = model(img).squeeze().cpu()
    epistemic, aleatoric = output.max(0)[0].mean().item(), abs(network.sigma.squeeze().item())
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


def stn_view(path, model, usecuda = True):
    with torch.no_grad():
      model.eval()
      img = Image.open(path)
      img = test_transform(img)
      if (usecuda):
        img = img.cuda()
      img = img.unsqueeze(0)
      output = model.stn(img).squeeze().cpu()
      return output.permute(1,2,0).numpy()
      #channel last numpy array hxwxc


def gradcam(path, model, usecuda = True):
  model.cam = True
  st = 0.7
  model.eval()
  img = Image.open(path)
  img = test_transform(img).unsqueeze(0)
  if usecuda:
    img = img.cuda()
  out = model(img)
  pred = out.argmax(1).squeeze()

  out[:,pred].backward()
  gradients = model.grad
  pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
  activations = model.activations(img).detach()
  for i in range(activations.shape[1]):
    activations[:, i, :, :] *= pooled_gradients[i]

  with torch.no_grad():
      heatmap = torch.mean(activations, dim=1).cpu().unsqueeze(1)
      heatmap = norm(torch.relu(heatmap));
      map = F.interpolate(heatmap,(img.shape[2],img.shape[2]),mode='bilinear')
      map = map.squeeze().unsqueeze(2); img = img.squeeze().cpu().permute(1,2,0)*255; map = np.uint8(255 * map)
      map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
      superimpose = (img + st*map)
      superimpose /= superimpose.max()
  model.cam = False
  return superimpose.numpy()
  #channel last numpy array hxwxc


def gradcam_noise(path, model, usecuda = True):
  model.cam = True
  st = 0.7
  model.eval()
  img = Image.open(path)
  img = test_transform(img).unsqueeze(0)
  if usecuda:
    img = img.cuda()
  out = model(img)
  pred = out.argmax(1).squeeze()

  model.sigma.backward()
  gradients = model.grad
  pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
  activations = model.activations(img).detach()
  for i in range(activations.shape[1]):
    activations[:, i, :, :] *= pooled_gradients[i]

  with torch.no_grad():
      heatmap = torch.mean(activations, dim=1).cpu().unsqueeze(1)
      heatmap = norm(torch.relu(heatmap));
      map = F.interpolate(heatmap,(img.shape[2],img.shape[2]),mode='bilinear')
      map = map.squeeze().unsqueeze(2); img = img.squeeze().cpu().permute(1,2,0)*255; map = np.uint8(255 * map)
      map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
      superimpose = (img + st*map)
      superimpose /= superimpose.max()
  model.cam = False
  return superimpose.numpy()
  #channel last numpy array hxwxc
