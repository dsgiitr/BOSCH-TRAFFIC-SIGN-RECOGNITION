import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
valid_df = []
test_df = []

# model should have model.datain, model.update_embeddings
def train(model, train_loader, epochs, cudav):

  learning_rate = 0.001
  momentum = 0.1
  weight_decay = 0.0001
  lm = 0.1
  log_interval = 20
  optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=learning_rate,weight_decay=weight_decay)

  for epoch in range(epochs):
    loss_=[]
    correct_=0

    for batch_idx, (loc, data, target) in enumerate(train_loader):
      data, target = Variable(data), Variable(target).long()
      if cudav :
        torch.cuda.empty_cache()
        data=data.cuda(); target=target.cuda();
      model.train()

      optimizer.zero_grad()
      output = model(data)
      pred = output.data.argmax(1).long()
      datain = model.datain
      loss = 10*F.binary_cross_entropy(output, F.one_hot(target,43).float()) + lm*calc_gradient_penalty(datain,output.sum(1))
      correct = pred.eq(target.data.view_as(pred)).sum().item()
      loss.backward()
      optimizer.step()

      loss_.append(loss.item())
      correct_+=correct

      with torch.no_grad():
        model.eval()
        model.update_embeddings(data,F.one_hot(target.long(),43).float())

      if batch_idx % log_interval == 0:
        trainlog(batch_idx,100.*correct/len(data),loss.item())

    trainlog(epoch,100.*correct_/len(train_loader.dataset),np.array(loss_).mean())

  if cudav:
    torch.cuda.empty_cache()




def validation(model,val_loader,cudav):
  model.eval()
  criterion = nn.CrossEntropyLoss()
  y_true = []
  y = []
  y_conf = []
  sig = []
  location = []
  val_loss_ = []
  correct_ = 0

  with torch.no_grad():
    for _,(loc, data, target) in enumerate(val_loader):
      target = target.long()
      if cudav:
        torch.cuda.empty_cache()
        data = data.cuda()
        target = target.cuda()

      output = model(data)
      pred = output.data.argmax(1).long()
      conf = output.data.max(1)[0]
      sigma = model.sigma

      val_loss = criterion(output, target.long()).item()
      correct = pred.eq(target.long().data.view_as(pred)).sum().item()
      val_loss_.append(val_loss)
      correct_+=correct

      y_true.append(target.cpu().numpy())
      y_conf.append(conf.cpu().numpy())
      y.append(pred.cpu().numpy())
      sig.append(sigma.cpu().numpy())
      location.append(np.array(loc))

  y_true = np.concatenate(y_true)
  y_conf = np.concatenate(y_conf)
  y = np.concatenate(y)
  sig = np.concatenate(sig)
  location = np.concatenate(location)
  data = np.column_stack([y_true, y, y_conf,sig, location])

  global valid_df
  valid_df = pd.DataFrame(data=data,columns=['label','prediction','confidence','sigma','location'])
  if cudav:
    torch.cuda.empty_cache()





def test(model,test_loader,cudav):
  model.eval()
  criterion = nn.CrossEntropyLoss()
  y_true = []
  y = []
  y_conf = []
  sig = []
  location = []
  test_loss_ = []
  correct_ = 0

  with torch.no_grad():
    for _,(loc, data, target) in enumerate(test_loader):
      target = target.long()
      if cudav:
        torch.cuda.empty_cache()
        data = data.cuda()
        target = target.cuda()

      output = model(data)
      pred = output.data.argmax(1).long()
      conf = output.data.max(1)[0]
      sigma = model.sigma

      test_loss = criterion(output, target.long()).item()
      correct = pred.eq(target.long().data.view_as(pred)).sum().item()
      test_loss_.append(test_loss)
      correct_+correct

      y_true.append(target.cpu().numpy())
      y_conf.append(conf.cpu().numpy())
      y.append(pred.cpu().numpy())
      sig.append(sigma.cpu().numpy())
      location.append(np.array(loc))

  y_true = np.concatenate(y_true)
  y_conf = np.concatenate(y_conf)
  y = np.concatenate(y)
  sig = np.concatenate(sig)
  location = np.concatenate(location)
  data = np.column_stack([y_true, y, y_conf,sig, location])

  global test_df
  test_df = pd.DataFrame(data=data,columns=['label','prediction','confidence','sigma','location'])
  if cudav:
    torch.cuda.empty_cache()


def trainlog(epoch, correct, loss):
  ;
