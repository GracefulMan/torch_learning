from glob import glob
import torch
import cv2
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
def data_loader():
    file_path = 'data/test_image/*.png'
    file_list =sorted(glob(file_path))
    test_data = np.empty((0, 1, 28, 28))
    for img_path in file_list:
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (28, 28))
        img = np.reshape(img,(1, 1, 28, 28))
        test_data = np.concatenate([test_data, img], axis=0)
    return torch.from_numpy(test_data/255.).type(torch.FloatTensor)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # 图片的channel，因为是灰度图，如果是rgb图，则是3，相当于输入filter的个数.
                out_channels=16,  # 输出filter的个数。同一个区域用16个filter进行卷积.
                kernel_size=5,
                stride=1,  # 步长
                padding=2  # 边缘填充0，如果stride = 1, padding =(kernel_size - 1)/2 = (5 - 1) / 2
            ),
            nn.ReLU(),  # ->(16, 28, 28)
            nn.MaxPool2d(kernel_size=2)  # ->(16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),  # 和上面顺序一致 # ->(32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2)  # ->(32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)  # (batch, 32, 7 ,7)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output

test_data = data_loader()
cnn_model = torch.load('cnn.pkl')
pred_y = cnn_model(test_data)
pred_y = torch.max(pred_y, 1)[1].data.numpy()
test_data =test_data.numpy()
test_data *= 255
test_data = np.reshape(test_data,(-1, 28, 28))
final_image = np.empty(shape=(28, 28 * 5))
for i in range(4):
    tmp = np.empty(shape=(28, 28))
    for j in range(4):
        tmp = np.concatenate([tmp, test_data[i * 4 + j]], axis= 1)
    final_image = np.concatenate([final_image, tmp], axis=0)


print(pred_y)
final_image = np.uint(final_image)
plt.imshow(final_image)
plt.show()