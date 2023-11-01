import os
import cv2
import glob
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary


class Question5:
    def __init__(self):
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # Load Weight
        self.checkpoint = torch.load('cifar10_vgg19.pth', map_location=torch.device('cpu'))
        self.model = torchvision.models.vgg19_bn(num_classes=10)
        self.model.load_state_dict(self.checkpoint)
        self.model.eval()
    
    # 5-1: Show augumentaion images
    def show_augumentation_imgs(self):
        # Data Augumentation
        transform = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),     # 隨機水平翻轉
            transforms.RandomCrop(32, padding=4),  # 隨機裁剪，可增加填充
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.015),  # 隨機顏色變換
            transforms.ToTensor()                  # 轉換至tensor
        ])
        
        files = glob.glob('./Q5_Image/Q5_1/*.png')

        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        
        for i, file in enumerate(files[:9]):
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transforms.ToPILImage()(img) 
            img = transform(img)

            img_np = img.permute(1, 2, 0).numpy()
            
            row, col = i // 3, i % 3
            axs[row, col].imshow(img_np)
            axs[row, col].axis('off')
            axs[row, col].set_title(os.path.splitext(os.path.basename(file))[0])

        plt.show()

    # 5-2: Show model structure
    def show_model_structure(self):
        model = torchvision.models.vgg19_bn(num_classes=10) 
        summary(model, (3, 32, 32))
    
    # 5-3: Show accuracy and loss curve
    def show_acc_loss(self):
        img = cv2.imread('Q5_image/Q5_3/loss_accuracy_curves.png',cv2.IMREAD_GRAYSCALE)  # queryImage
        cv2.imshow('Accuracy and Loss', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 5-4: Inference
    def inference(self, img_path):
        img = np.array(plt.imread(img_path)).astype(np.uint8)
        img = np.array(Image.fromarray(img).resize((32, 32))) # (3, 32, 32)
        img_batch = img.reshape(-1, 3, 32, 32)
        img_tensor = torch.from_numpy(img_batch).float()

        output = self.model(img_tensor.to(self.device))
        _, predicted = torch.max(output.data, 1)
        pre = predicted.cpu().numpy()
        
        print(self.classes[pre[0]])

        # 將輸出轉換成機率
        probabilities = nn.functional.softmax(output[0], dim=0)

        plt.figure(figsize=(8, 6))
        plt.bar(self.classes, probabilities.cpu().detach().numpy())
        plt.xticks(rotation=45)
        plt.ylabel('Probability')
        plt.title('Probability of each class')
        plt.show()
        
        return self.classes[pre[0]]
