import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import pandas as pd
import numpy as np
import utils.dataset_loader as dl
from flask import current_app
import subprocess
valid_df = []
v_logit = []
model = []
url = "http://localhost:6006/"
#use_gpu = torch.cuda.is_available()
use_gpu = False
img_size = 40
hidden = 0
path = os.getcwd()
folder = 'tensorboard'
writer = SummaryWriter(folder)
completed = "false"
n_classes = 0


def trainlog(name, epoch, correct, loss):
    print("name = {}, epoch = {}, correct = {}, loss = {}".format(
        name, epoch, correct, loss))


def calc_gradient_penalty(x, y_pred_sum):
    gradients = torch.autograd.grad(
        outputs=y_pred_sum,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred_sum),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.flatten(start_dim=1)

    # L2 norm
    grad_norm = gradients.norm(2, dim=1)

    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()

    return gradient_penalty

# model should have model.datain, model.update_embeddings


def train(model, n_classes, train_loader, val_loader, lr, lam, weight_d, epochs, opt, cudav):

    learning_rate = lr
    momentum = 0.1
    weight_decay = weight_d
    lm = lam
    log_interval = 20
    if opt == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters(
        )), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters(
        )), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [8, 10, 14], 0.5)

    for epoch in range(epochs):
        current_app.logger.info(epoch)
        loss_ = []
        correct_ = 0

        for batch_idx, (loc, data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target).long()
            if cudav:
                torch.cuda.empty_cache()
                data = data.cuda()
                target = target.cuda()
            model.train()

            optimizer.zero_grad()
            output = model(data)
            pred = output.data.argmax(1).long()
            datain = model.datain
            loss = 10*F.binary_cross_entropy(output, F.one_hot(
                target, n_classes).float()) + lm*calc_gradient_penalty(datain, output.sum(1))
            correct = pred.eq(target.data.view_as(pred)).sum().item()
            loss.backward()
            optimizer.step()

            loss_.append(loss.item())
            correct_ += correct

            with torch.no_grad():
                model.eval()
                model.update_embeddings(data, F.one_hot(
                    target.long(), n_classes).float())

            if batch_idx % log_interval == 0:
                trainlog('batch', batch_idx, 100. *
                         correct/len(data), loss.item())
                writer.add_scalar('Batch / Training loss',
                                  loss.item(),
                                  epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Batch / Training Correct',
                                  100.*correct/len(data),
                                  epoch * len(train_loader) + batch_idx)

        v_correct, v_loss = validation(model, val_loader, cudav)
        trainlog('epoch complete', epoch, 100.*correct_ /
                 len(train_loader.dataset), np.array(loss_).mean())
        writer.add_scalars('Epoch / Loss',
                           {"train": np.array(loss_).mean(),
                            "validation": v_loss}, epoch)
        writer.add_scalars('Epoch / Correct',
                           {"train": 100.*correct_/len(train_loader.dataset),
                            "validation": v_correct}, epoch)
        scheduler.step()
    if cudav:
        torch.cuda.empty_cache()


def validation(model, val_loader, cudav):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    global v_logit
    y_true = []
    y = []
    y_conf = []
    v_logit = []
    sig = []
    location = []
    val_loss_ = []
    correct_ = 0

    with torch.no_grad():
        for _, (loc, data, target) in enumerate(val_loader):
            current_app.logger.info("starting validation")
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
            correct_ += correct

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
    data = np.column_stack([y_true, y, y_conf, sig, location])

    global valid_df
    valid_df = pd.DataFrame(
        data=data, columns=['label', 'prediction', 'confidence', 'sigma', 'location'])
    if cudav:
        torch.cuda.empty_cache()

    trainlog('validation', 1, 100.*correct_ /
             len(val_loader.dataset), np.array(val_loss_).mean())
    return 100.*correct_/len(val_loader.dataset), np.array(val_loss_).mean()


# def test(model,test_loader,cudav):
#   model.eval()
#   criterion = nn.CrossEntropyLoss()
#   global t_logit
#   y_true = []
#   y = []
#   y_conf = []
#   t_logit = []
#   sig = []
#   location = []
#   test_loss_ = []
#   correct_ = 0

#   with torch.no_grad():
#     for _,(loc, data, target) in enumerate(test_loader):
#       target = target.long()
#       if cudav:
#         torch.cuda.empty_cache()
#         data = data.cuda()
#         target = target.cuda()

#       output = model(data)
#       pred = output.data.argmax(1).long()
#       conf = output.data.max(1)[0]
#       sigma = model.sigma

#       test_loss = criterion(output, target.long()).item()
#       correct = pred.eq(target.long().data.view_as(pred)).sum().item()
#       test_loss_.append(test_loss)
#       correct_+correct

#       y_true.append(target.cpu().numpy())
#       y_conf.append(conf.cpu().numpy())
#       y.append(pred.cpu().numpy())
#       t_logit.append(output.cpu().numpy())
#       sig.append(sigma.cpu().numpy())
#       location.append(np.array(loc))

#   y_true = np.concatenate(y_true)
#   y_conf = np.concatenate(y_conf)
#   y = np.concatenate(y)
#   t_logit = np.concatenate(t_logit)
#   sig = np.concatenate(sig)
#   location = np.concatenate(location)
#   data = np.column_stack([y_true, y, y_conf,sig, location])

#   global test_df
#   test_df = pd.DataFrame(data=data,columns=['label','prediction','confidence','sigma','location'])
#   if cudav:
#     torch.cuda.empty_cache()


# def trainlog(message,epoch, correct, loss):
#     print(message + " epoch = {}, correct = {}, loss = {}".format(epoch,correct,loss))


def runtraining(layers, epochs=15, batch_size=64, learning_rate=0.0003, centroid_size=100, lm=0.1, weight_decay=0.0001, opt="Adam"):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    # tensorboard
    # os.system("tensorboard --reload_interval 15 --logdir tensorboard")
    tensorboard_proc = subprocess.Popen(
        ["tensorboard", "--reload_interval", "15", "--logdir", folder])
    current_app.logger.info("tensorflow launched \n")
    global completed
    completed = "true"

    global model, writer, use_gpu, n_classes

    root_dir = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(root_dir, '..', 'data', 'split', 'train')
    valid_path = os.path.join(root_dir, '..', 'data', 'split', 'valid')
    n_classes = len(dl.find_classes(train_path)[0])
    train_loader = dl.create_loader(
        train_path, batch_size=batch_size, shuffle=True,  sz=img_size)
    valid_loader = dl.create_loader(
        valid_path, batch_size=batch_size, shuffle=False, sz=img_size)

    model = makemodel(layers, n_classes, centroid_size)
    current_app.logger.info("model is made \n")

    for _, data, target in train_loader:
        writer.add_embedding(
            data.view(data.shape[0], -1), metadata=target, label_img=data)
        writer.add_graph(model.cpu(), data)
        break
    if use_gpu:
        model.cuda()
    current_app.logger.info("about to start training \n")
    train(model, n_classes, train_loader, valid_loader,
          learning_rate, lm, weight_decay, epochs, opt, use_gpu)
    validation(model, valid_loader, use_gpu)


def makemodel(layers, n_classes, embedding_size):
    global hidden
    modules = []
    hidden = 0
    for layer in layers:
        if layer == "Conv":
            hidden += 1
            modules.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        elif layer == "B_Norm":
            modules.append(nn.BatchNorm2d(128))
        elif layer == "Relu":
            modules.append(nn.ReLU(inplace=True))

    print(modules)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.grad = None
            self.cam = False
            # CNN layers
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),

                nn.Conv2d(32, 32, kernel_size=3),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),

                nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=14),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),

                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25)
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 128, kernel_size=3),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, 128, kernel_size=3),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=6),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.25),

            )

            self.conv25 = nn.Sequential(*modules)

            self.conv3 = nn.Sequential(
                nn.AvgPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=5, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.5)
            )

            self.fc = nn.Sequential(
                #nn.Linear(128 * 1 * 1, 128),
            )

            self.localization = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(8, 10, kernel_size=5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )

            # Regressor for the 3 * 2 affine matrix
            self.fc_loc = nn.Sequential(
                nn.Linear(10 * 6 * 6, img_size),
                nn.ReLU(True),
                nn.Linear(img_size, 3 * 2),
            )

            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor(
                [1, 0, 0, 0, 1, 0], dtype=torch.float))

        # For Grad
        def activations_hook(self, grad):
            self.grad = grad

        def activations(self, x):
            x = self.stn(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv25(x)
            return x

        # Spatial transformer network forward function
        def stn(self, x):
            xs = self.localization(x)
            xs = xs.view(-1, 10 * 6 * 6)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)
            grid = F.affine_grid(theta, x.size())
            x = F.grid_sample(x, grid)
            return x

        def compute_features(self, x):
            x = self.stn(x)
            self.datain = x
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv25(x)

            if self.cam:
                h = x.register_hook(self.activations_hook)

            x = self.conv3(x)
            x = x.view(-1, 256 * 3 * 3)
            x = self.fc(x)
            return x

    class CNN_DUQ(Net):
        def __init__(
            self,
            input_size,
            num_classes,
            centroid_size,
            model_output_size,
            length_scale,
            gamma,
            input_dep_ls=False
        ):
            super().__init__()

            self.gamma = gamma
            self.input_dep_ls = input_dep_ls
            self.W = nn.Parameter(
                torch.zeros(centroid_size, num_classes, model_output_size))
            nn.init.kaiming_normal_(self.W, nonlinearity="relu")
            self.register_buffer("N", torch.zeros(num_classes) + 13)
            self.register_buffer(
                "m", torch.normal(torch.zeros(
                    centroid_size, num_classes), 0.05)
            )
            self.m = self.m * self.N
            if input_dep_ls:
                self.sigmann = nn.Sequential(
                    nn.Linear(model_output_size, 100),
                    nn.ReLU(True),
                    nn.Dropout2d(0.35),
                    nn.Linear(100, 1))
            self.sigma = length_scale*length_scale

        def rbf(self, z):
            if self.input_dep_ls:
                self.sigma = torch.sigmoid(self.sigmann(z)/50)+0.001
            z = torch.einsum("ij,mnj->imn", z, self.W)
            embeddings = self.m / self.N.unsqueeze(0)
            diff = z - embeddings.unsqueeze(0)
            diff = (- diff ** 2).mean(1).div(2 * (self.sigma)).exp()
            return diff

        def update_embeddings(self, x, y):
            self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)
            z = self.compute_features(x)
            z = torch.einsum("ij,mnj->imn", z, self.W)
            embedding_sum = torch.einsum("ijk,ik->jk", z, y)
            self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum

        def forward(self, x):
            z = self.compute_features(x)
            y_pred = self.rbf(z)
            return y_pred

    main_model = CNN_DUQ(32, n_classes, embedding_size,
                         256*3*3, 0.6, 0.999, True).float()
    if use_gpu:
        main_model.cuda()
    return main_model
