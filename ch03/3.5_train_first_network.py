import torch
import torch.nn as nn
import torchvision
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 16 * 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16 * 16, 10)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # N 1 28 28
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        out = self.sig(x)
        return out


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1)),
        torchvision.transforms.RandomRotation((-5, 5)),
        torchvision.transforms.ColorJitter(brightness=0.3,
                                           contrast=0.3),
        torchvision.transforms.CenterCrop(
            (int(28 * 0.8), int(28 * 0.8))),
        torchvision.transforms.Resize((28, 28)),
        torchvision.transforms.ToTensor(),
    ]
)


def train():
    model = MLP()
    model.eval()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)

    train_dataset = torchvision.datasets.MNIST(root="./",
                                               download=True,
                                               train=True,
                                               transform=torchvision.
                                               transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root="./",
                                              download=True,
                                              train=False,
                                              transform=torchvision.
                                              transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=8,
                              shuffle=True, num_workers=0,
                              drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=8,
                             shuffle=True, num_workers=0,
                             drop_last=True)
    criteria = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-2,
                    momentum=0.9, weight_decay=1e-4)

    def lr_step_func(epoch):
        return 1.0 if epoch < 1 else 0.1 ** len(
            [m for m in [10, 15, 18] if m - 1 <= epoch])

    scheduler_lr = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer, lr_lambda=lr_step_func)

    whole_train_steps = len(train_loader)
    for epoch in range(20):
        model.train()
        Loss_Sum = 0.0
        print("epoch：{} using lr :{} ".format(epoch,
                                              optimizer.param_groups
                                              [0]['lr']))
        for cnt, (imgs, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            out = model(imgs)
            loss = criteria(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss_Sum += float(loss)

            # print("epoch: {} [{}/{}] "
            #       "loss:{} Acc:{}".format(epoch, cnt,
            #                               len(train_loader),
            #                               float(loss),
            #                               float((torch.argmax(
            #                                   out, dim=1)
            #                                      == labels).sum())
            #                               / float(imgs.shape[0])))
        print("epoch：{} Loss_Sum:{} lr:{}".format(epoch, Loss_Sum
                                                  / whole_train_steps,
                                                  scheduler_lr.get_lr()
                                                  ))
        scheduler_lr.step()
        model.eval()
        AllCnt = 0.0
        RightCnt = 0.0
        for cnt, (imgs, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            out = model(imgs)
            AllCnt += imgs.shape[0]
            RightCnt += (torch.argmax(out, dim=1) == labels).sum()
        print("epoch：{} TestAcc:{}".format(epoch, RightCnt / AllCnt))
    torch.save(model.state_dict(), "epoch_{}.pth".format(epoch))


if __name__ == '__main__':
    train()
