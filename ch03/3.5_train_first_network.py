import torch
import torch.nn as nn
import torchvision
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
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


def case_infer():
    input = torch.rand([5, 1, 28, 28])
    model = MLP()
    model.eval()

    out = model(input)
    print(out.shape)


def case_train():
    model = MLP()
    model.eval()

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
    criteria = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(20):
        model.train()
        Loss_Sum = 0.0
        for cnt, (imgs, labels) in enumerate(train_loader):
            out = model(imgs)
            loss = criteria(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss_Sum += float(loss)
            print("epoch: {} [{}/{}] "
                  "loss:{} Acc:{}".format(epoch, cnt,
                                          len(train_loader),
                                          float(loss),
                                          float((torch.argmax(
                                              out, dim=1)
                                                 == labels).sum())
                                          / float(imgs.shape[0])))
        print("epoch：{} Loss_Sum:{}".format(epoch, Loss_Sum
                                            / len(train_loader)))
        model.eval()
        AllCnt = 0.0
        RightCnt = 0.0
        for cnt, (imgs, labels) in enumerate(test_dataset):
            out = model(imgs)
            AllCnt += imgs.shape[0]
            RightCnt += (torch.argmax(out, dim=1) == labels).sum()
        print("epoch：{} TestAcc:{}".format(epoch, RightCnt / AllCnt))
        torch.save(model.state_dict(), "epoch_{}.pth".format(epoch))


if __name__ == '__main__':
    # case_infer()
    case_train()
