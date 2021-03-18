import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import os, shutil
import pandas as pd
import numpy as np

valid_df = []
test_df = []
v_logit = []
t_logit = []
model = []
url = "http://localhost:6006/"
path = os.getcwd()
img_size = 40
writer = SummaryWriter('tensorboard')
train_loader = []
val_loader = []
test_loader = []

# model should have model.datain, model.update_embeddings
def train(model, train_loader,val_loader,lr,lam,weight_d, epochs,opt, cudav):

  learning_rate = lr
  momentum = 0.1
  weight_decay = weight_d
  lm = lam
  log_interval = 20
  if opt == "Adam":
      optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=learning_rate,weight_decay=weight_decay)
  else:
      optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()),lr=learning_rate,weight_decay=weight_decay)


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
        writer.add_scalar('Batch / Training loss',
                            loss.item(),
                            epoch * len(train_loader) + batch_idx)
        writer.add_scalar('Batch / Correct',
                            100.*correct/len(data),
                            epoch * len(train_loader) + batch_idx)

    v_correct, v_loss = validation(model, val_loader,True)
    trainlog(epoch,100.*correct_/len(train_loader.dataset),np.array(loss_).mean())
    writer.add_scalars('Epoch / Loss',
                      {"train":np.array(loss_).mean(),
                        "validation": v_loss},epoch)
    writer.add_scalars('Epoch / Correct',
                      {"train":100.*correct_/len(train_loader.dataset),
                        "validation": v_correct},epoch )
    scheduler.step()
  if cudav:
    torch.cuda.empty_cache()




def validation(model,val_loader,cudav):
  model.eval()
  criterion = nn.CrossEntropyLoss()
  y_true = []
  y = []
  y_conf = []
  v_logit = []
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
      v_logit.append(output.cpu().numpy())
      sig.append(sigma.cpu().numpy())
      location.append(np.array(loc))

  y_true = np.concatenate(y_true)
  y_conf = np.concatenate(y_conf)
  y = np.concatenate(y)
  v_logit = np.concatenate(v_logit)
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
  t_logit = []
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
      t_logit.append(output.cpu().numpy())
      sig.append(sigma.cpu().numpy())
      location.append(np.array(loc))

  y_true = np.concatenate(y_true)
  y_conf = np.concatenate(y_conf)
  y = np.concatenate(y)
  t_logit = np.concatenate(t_logit)
  sig = np.concatenate(sig)
  location = np.concatenate(location)
  data = np.column_stack([y_true, y, y_conf,sig, location])

  global test_df
  test_df = pd.DataFrame(data=data,columns=['label','prediction','confidence','sigma','location'])
  if cudav:
    torch.cuda.empty_cache()

def trainlog(epoch, correct, loss):
    ;
# tensorboard
os.system("tensorboard --logdir tensorboard")



def runtraining(epochs = 15, batch_size = 64,learning_rate = 0.0003,centroid_size = 100,lm = 0.1,weight_decay = 0.0001,opt = "Adam"):
    if(os.path.exists(os.path.join(path,'tensorboard'))):
        shutil.rmtree(os.path.join(path,'tensorboard'))
    os.mkdir(os.path.join(path,'tensorboard'))
    use_gpu = True
    train_loader,val_loader = create_loader('./data/split/train', batch_size=batch_size, shuffle=True, cudav = use_gpu, split = 0.1,sz = img_size)
    test_loader = create_loader('./data/split/test', batch_size=batch_size, shuffle=False,  cudav = use_gpu, sz = img_size)
    _,data,target = next(iter(train_loader))

    writer.add_embedding(data.view(batch_size,-1),metadata=target,label_img=data)
    writer.add_graph(model.cpu(), data)
    train(model,train_loader,val_loader,learning_rate,lm,weight_decay, epochs, opt, use_gpu)
    validation(model,val_loader,use_gpu)
    test(model,test_loader,use_gpu)

def makemodel():
    ;
