import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data
# hyper-parameters
EPOCH = 5
BATCH_SIZE = 64
LR = 0.01
DOWNLOAD_DATASET = False
TIME_STEP = 28
INPUT_SIZE = 28
HIDDEN_SIZE = 64

train_data = dsets.MNIST(root='./data/mnist',train=True, transform=transforms.ToTensor(), download=False)
train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=False, num_workers=4)
test_data = dsets.MNIST(root='./data/mnist', train=False, transform=transforms.ToTensor())
test_x = Variable(test_data.data).type(torch.FloatTensor) / 255.
test_y = test_data.targets

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True # indicate whether the input shape is (batch_size, :, :)
        )
        self.out = torch.nn.Linear(
            in_features=HIDDEN_SIZE,
            out_features=10
        )

    def forward(self, x):
        x, (h_n, h_c) = self.rnn(x, None)
        return self.out(x[:, -1, :]) # (batch, time_step, input)


my_rnn = RNN()
optimizer = torch.optim.Adam(my_rnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_pred_y = my_rnn(batch_x.view(-1, 28, 28))
        loss = loss_func(batch_pred_y, batch_y)
        if step % 5 == 0:
            pred_y = my_rnn(test_x.view(-1, 28, 28))
            pred_y = torch.max(pred_y, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / (test_y.size(0) + 1e-10)
            print('Epoch: ',epoch," |train loss:%.4f" % loss, " |test accuracy: %.4f" % accuracy)


test_output = my_rnn(test_x[:10].view(-1, 28, 28)) # (batch, time_step, input)
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y,' prediction number')
print(test_y[:10], ' real number')
