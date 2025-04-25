import time
import torch
import wandb
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import cProfile


class MySampler(torch.utils.data.Sampler):  # Custom sampler for balancing classes in each batch
    def __init__(self, data_source):
        self.data_source = np.array(data_source)
        self.classes = np.unique(data_source)
        self.indexes = []
        self.arrays = []
        self.using = []
        for c in self.classes:
            self.arrays.append((self.data_source == c).nonzero()[0])  # get indexes of images from each class
            cop = self.arrays[-1].copy()  # copy indexes
            np.random.shuffle(cop)  # shuffle indexes in one class
            self.using.append(cop)  # add indexes, which are being used
            self.indexes.append(0)  # append 0 to indicate the next element for this class is 0

    def __iter__(self):
        i = 0
        while i < len(self):  # stop function for epoch
            for c in self.classes:
                yield self.using[c][self.indexes[c]]  # return element according to index
                i += 1  # increment number of images used this epoch
                self.indexes[c] += 1  # increment index of image
                if self.indexes[c] == len(self.using[c]):  # reshuffle indexes, if all were used for that class
                    np.random.shuffle(self.using[c])
                    self.indexes[c] = 0  # reset index to 0

    def __len__(self):
        return len(self.data_source)  # return the length of dataset and epoch


def getMNIST(batch_size=256):
    transform = transforms.Compose([  # Transform for train set
        transforms.Pad(2),  # Padding to introduce small translation of image
        transforms.RandomCrop((28, 28)),  # Crop to get to original size
        transforms.ToTensor(),
        transforms.Normalize((0.13066047,), (0.30810780,))
    ])
    ttM = transforms.Compose([  # Transform for test set
        transforms.ToTensor(),
        transforms.Normalize((0.13066047,), (0.30810780,))
    ])
    trainset = datasets.MNIST('data/', train=True, download=True, transform=transform)
    testset = datasets.MNIST('data/', train=False, download=True, transform=ttM)
    lab = trainset.targets  # creation of sampler to make classes in batches more balanced
    sampler_init = MySampler(lab)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2,
                                              sampler=sampler_init)  # loader for the training set
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=2)  # loader for the testing set
    return {"train": trainloader, "test": testloader}  # set of loaders


def getCIFAR(batch_size=256):
    transform = transforms.Compose([
        transforms.Pad(3),
        transforms.RandomCrop((28, 28)),
        transforms.ToTensor()
    ])
    trainset = datasets.CIFAR10('data/', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10('data/', train=False, download=True, transform=transforms.ToTensor())
    lab = trainset.targets  # creation of sampler to make classes in batches more balanced
    sampler_init = MySampler(lab)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=2, sampler=sampler_init)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return {"train": trainloader, "test": testloader}


def convol(in_channel, out_channel, kernel):  # method for easier creation of convolutional blocks
    return nn.Sequential(  # each block has one convolution, batchnorm, relu and maxpool
        nn.Conv2d(in_channel, out_channel, kernel, padding="valid")
        , nn.BatchNorm2d(out_channel)
        , nn.ReLU()
        , nn.MaxPool2d(2)
    )


def squash(s):  # original squash function from Sara Sabour
    s2 = (s ** 2).sum(dim=-1, keepdims=True)
    return s2 / (1 + s2) * s / torch.sqrt(s2 + 1e-8)


class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, kernel, capsules, cap_dim, chan=18):  # decrease chan
        super(PrimaryCaps,
              self).__init__()  # Primary caps layer, which uses one convolution and reshapes the data into capsules
        self.capsules = capsules
        self.cap_dim = cap_dim
        self.conv = nn.Sequential(nn.Conv2d(in_channels, chan * cap_dim, kernel, stride=2, padding="valid"), nn.ReLU())

    def forward(self, x):  # x = [batch, channel, width, height]
        x = self.conv(x).transpose(1,
                                   3)  # switching dimensions to get the same shape as in tensorflow  x = [batch,
        # height, width, channel]
        x = squash(x.reshape(x.shape[0], self.capsules, self.cap_dim))  # squash and reshape to needed dimensions
        return x  # x = [batch, num_caps, caps_dim]


class RoutingLayer(nn.Module):
    def __init__(self, caps_in, caps_out, size_in, size_out, iterations=3):
        super().__init__()
        self.caps_in, self.caps_out = caps_in, caps_out
        self.weight = nn.Parameter(torch.randn(caps_in, caps_out, size_in, size_out), requires_grad=True)
        self.r = iterations
        self.sfmx = nn.Softmax(dim=1)
        self.last_iteration_c = torch.zeros((1, caps_in, caps_out), requires_grad=False)

    def forward(self, u):  # u = [batch, num_caps_in, caps_dim_in]
        u_hat = torch.einsum('ijnm, bin->bijm', self.weight,
                             u)  # u_hat = [batch, num_caps_in, num_caps_out, caps_dim_out]
        b = u.new_zeros(u.shape[0], self.caps_in, self.caps_out)  # b = [batch, num_caps_in, num_caps_out]

        for i in range(self.r):
            c = self.sfmx(b)  # c = [batch, num_caps_in, num_caps_out]
            v = squash(torch.einsum('bij,bijm->bjm', c, u_hat))  # v = [batch, num_caps_out, caps_dim_out]

            if i < self.r - 1:
                a = torch.einsum('bjm,bijm->bij', v, u_hat)  # a = [batch, num_caps_in, num_caps_out]
                b = b + a
            else:
                self.last_iteration_c = c.clone()

            # print(f"Iteration {i}:")
            # print(f"b: {b}")
            # print(f"c: {c}")
            # print(f"Sum of routing weights (c.sum(dim=1)): {c.sum(dim=1)}")
            # print("--------")
            #
            # print(f"Shape of u: {u.shape}")
            # print(f"Shape of u_hat: {u_hat.shape}")
            # print(f"Shape of v: {v.shape}")
            # print("========\n")

        return v  # v = [batch, num_caps_out, caps_dim_out]


class MarginLoss(nn.Module):
    def __init__(self, n_classes, lambda_=0.5, m_positive=0.9, m_negative=0.1):
        super().__init__()
        self.m_negative = m_negative
        self.m_positive = m_positive
        self.lambda_ = lambda_
        self.n_classes = n_classes

    def forward(self, v: torch.Tensor, labels: torch.Tensor):
        v = torch.sqrt((v ** 2).sum(dim=-1))
        labels = F.one_hot(labels, num_classes=self.n_classes)
        loss = labels * F.relu(self.m_positive - v) + self.lambda_ * (1.0 - labels) * F.relu(v - self.m_negative)
        # print(loss)
        return loss.sum(dim=-1).mean()


class CapsuleModel(nn.Module):
    def __init__(self, conv, capdim, in_channels):
        super(CapsuleModel, self).__init__()
        self.classes = capdim[-1][0]
        convs = [convol(in_channels, conv[0], 3)]
        for i in range(len(conv) - 1):
            convs.append(convol(conv[i], conv[i + 1], 3))
        self.conv1 = nn.Sequential(*convs)
        self.primaryCaps = PrimaryCaps(conv[-1], 3, capdim[0][0], capdim[0][1])
        routings = [RoutingLayer(capdim[i][0], capdim[i + 1][0], capdim[i][1], capdim[i + 1][1]) for i in range(len(capdim) - 2)]
        self.routing = nn.Sequential(*routings)
        self.dropout = nn.Dropout(0.2)
        self.routing_final = RoutingLayer(capdim[-2][0], capdim[-1][0], capdim[-2][1], capdim[-1][1])
        self.number_of_capsules_from_first_layer = capdim[0][0]
        self.activations_from_first_layer = []
        self.primary_feature_map = None
        self.primary_caps_spatial_shape = None

    def forward(self, x):
        self.activations_from_first_layer = []
        x_conv = self.conv1(x)
        fmap = self.primaryCaps.conv(x_conv)
        self.primary_feature_map = fmap.detach().cpu()
        B, C, H, W = fmap.shape
        self.primary_caps_spatial_shape = (H, W)
        caps = fmap.permute(0, 2, 3, 1).contiguous().view(B, -1, self.primaryCaps.cap_dim)
        caps = squash(caps)
        self.activations_from_first_layer = caps[:, :self.number_of_capsules_from_first_layer, :].detach()
        caps = self.routing(caps)
        caps = self.dropout(caps)
        caps = self.routing_final(caps)
        return caps

    def get_feature_map(self):
        return self.primary_feature_map

    def get_caps_spatial_shape(self):
        return self.primary_caps_spatial_shape

    def get_activations(self):
        return self.activations_from_first_layer

    def getC(self, x):
        caps = self.conv1(x)  # Feature extraction
        caps = self.primaryCaps(caps)  # Primary caps layer

        all_last_iteration_coefficients = []

        # For each routing layer
        for routing_layer in self.routing:
            caps = routing_layer(caps)
            all_last_iteration_coefficients.append(routing_layer.last_iteration_c)

        caps = self.routing_final(caps)  # final routing layer
        all_last_iteration_coefficients.append(self.routing_final.last_iteration_c)

        # print("All last iteration: ", all_last_iteration_coefficients)

        return all_last_iteration_coefficients


class Recon(nn.Module):
    def __init__(self, capdim, out_channels):
        super(Recon, self).__init__()
        self.classes = capdim[0]
        self.reconstruction = nn.Sequential(  # Simple Linear reconstructor
            nn.Linear(capdim[0] * capdim[1], 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28 * out_channels),
            nn.Sigmoid()
        )
        self.out_channels = out_channels

    def forward(self, x, y=None):
        with torch.no_grad():
            if y is None:  # getting predicted class if not defined
                y = (x ** 2).sum(-1).argmax(-1)
            mask = F.one_hot(y, num_classes=self.classes).unsqueeze(-1)
        recon = (x * mask).view(x.shape[0], -1)  # masking the output of classification
        recon = self.reconstruction(recon)
        recon = recon.view(-1, self.out_channels, 28, 28)
        return recon


class ReconConv(nn.Module):
    def __init__(self, capdim, out_channels):
        super(ReconConv, self).__init__()
        self.classes = capdim[0]
        self.reconLinear = nn.Linear(capdim[0] * capdim[1], 12 * 5 * 5)
        self.reconstruction = nn.Sequential(  # More complicated convolutional reconstructor
            nn.ConvTranspose2d(12, 12, 5, stride=2),
            # Number and padding of convtranspose layers can be changed according to need
            nn.Conv2d(12, 6, 3, padding="same"),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 6, 3, stride=2, output_padding=1),
            nn.Conv2d(6, out_channels, 3, padding="same"),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        with torch.no_grad():
            if y is None:  # getting predicted class if not defined
                y = (x ** 2).sum(-1).argmax(-1)
            mask = F.one_hot(y, num_classes=self.classes).unsqueeze(-1)
        recon = (x * mask).view(x.shape[0], -1)  # masking the output of classification
        recon = self.reconLinear(recon)
        recon = recon.view(-1, 12, 5, 5)
        recon = self.reconstruction(recon)
        return recon


def train(lr=0.0035, coef=0.7, conv_size=None, capdim=None):
    if conv_size is None:
        conv_size = [8, 16]
    if capdim is None:
        capdim = [(72, 4), (20, 15), (10, 20)]  # capdim = [(72, 4), (20, 15), (10, 20)]
    loaders = getMNIST()  # loader for dataset
    in_channels = 1  # number of channels of input images
    # recon = Recon(capdim=capdim[-1], out_channels=in_channels).to(device)  # simple recon
    recon = ReconConv(capdim=capdim[-1], out_channels=in_channels).to(device)  # convolutional recon
    model = CapsuleModel(conv=conv_size, capdim=capdim, in_channels=in_channels).to(device)  # capsule model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer
    lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                      gamma=0.98)  # weight decay, which is used in capsules
    criterion = MarginLoss(capdim[-1][0]).to(device)  # loss function for classification
    rcriterion = nn.MSELoss().to(device)  # loss function for reconstruction
    pp = sum(p.numel() for p in model.parameters() if p.requires_grad)  # calculation of the number of parameters
    wandb.init(
        # set the wandb project where this run will be logged
        project="capsule-neural-zp",

        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "coef": coef,
            "Parameters": pp,
            "Model": model,
            "Convs": conv_size,
            "Caps": capdim
        }
    )
    val_acc_max = 0  # variable to keep the performance of the best model
    for epoch in range(100):
        start = time.time()
        model.train()  # change model to train mode
        ls = []  # empty list to save classification loss
        for inputs, labels in loaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            caps = model(inputs)  # get output from capsule model.  Output is [batch, classes, caps_dim]
            recons = recon(caps, labels)  # get output from reconstruction.   Output is [batch, channel, height, width]
            l1 = criterion(caps, labels)  # classification error
            l2 = rcriterion(recons, inputs)  # reconstruction error
            loss = l1 + coef * l2  # final loss
            loss.backward()  # calculating gradients
            ls.append(l1.detach().cpu().item())
            optimizer.step()  # changing weights
            optimizer.zero_grad()
        lr_decay.step()  # learning rate decayed by gamma: lr = lr * gamma
        ls = sum(ls) / len(ls)
        val_loss = []
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in loaders["test"]:
                inputs, labels = inputs.to(device), labels.to(device)
                caps = model(inputs)
                loss = criterion(caps, labels)
                val_loss.append(loss.detach().cpu().item())
                labels, preds = labels.cpu(), (caps ** 2).sum(dim=-1).argmax(-1).cpu()
                correct += (labels == preds).sum()
                total += len(labels)
        acc = correct / total
        if val_acc_max < acc:
            val_acc_max = acc
            torch.save(model.state_dict(), "capsnet_best.pth")
        val_loss = sum(val_loss) / len(val_loss)
        print(
            "Epoch %d, train_loss %4.4f, val_loss %4.4f, acc %4.4f, time %4.2f" % (
                epoch, ls, val_loss, acc, time.time() - start))
        wandb.log({"acc": acc, "train_loss": ls, "val_loss": val_loss, "time": time.time() - start})


if __name__ == '__main__':
    wandb.login(key="9c3a61075946169850c5197b8f5ddcb98b0ebcb9")
    device = "cuda:0"
    try:
        cProfile.run("train()", sort='cumulative')
    except Exception as e:
        print(f"An error occurred: {e}")
    wandb.finish()
