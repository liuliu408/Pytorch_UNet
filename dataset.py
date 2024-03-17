import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((480,480)),        # 缩放图像
    transforms.ToTensor(),               # 转为Tensor
])


# 数据处理文件
class Data_Loader(Dataset):          # 加载数据
    def __init__(self, root, transforms = transform):               # 指定路径、预处理等等
        imgs = os.listdir(root)                                     # 获取root文件下的文件
        self.imgs = [os.path.join(root,img) for img in imgs]        # 获取每个文件的路径
        self.transforms = transforms                                # 预处理

    def __getitem__(self, index):    # 读取图片，返回一条样本
        image_path = self.imgs[index]                       # 根据index读取图片
        label_path = image_path.replace('image', 'label')   # 把路径中的image替换成label，就找到对应数据的label

        image = Image.open(image_path)                      # 读取图片和对应的label图
        label = Image.open(label_path)

        if self.transforms:                                 # 判断是否预处理
            image = self.transforms(image)

            label = self.transforms(label)
            label[label>=0.5] = 1               # 这里转为二值图片
            label[label< 0.5] = 0

        return image, label

    def __len__(self):  # 返回样本的数量
        return len(self.imgs)


# if __name__ == "__main__":
#
#     dataset = Data_Loader("./data/test/image")               # 加载数据
#     print(len(dataset))           # 样本总数：21
#     for image,label in dataset:
#         print(image)
#         print('image size:',image.size())   # image size: torch.Size([1, 480, 480])
#         print(label)
#         print('label size:',label.size())   # label size: torch.Size([1, 480, 480])
#         break
