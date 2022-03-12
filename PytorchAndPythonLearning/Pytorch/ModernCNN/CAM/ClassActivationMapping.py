# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# last update by BZ, June 30, 2021

"""
对于全卷积网络，提取的特征在空间上还没有被破坏，
也就是说，目标定位的能力还没有失去，
这时可以通过最后一层softmax分类器的权重来是它可视化，这是attention的前身 

"""

import io
from PIL import Image
from torch import Tensor
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

# input image
LABELS_file = r'PytorchAndPythonLearning\Pytorch\ModernCNN\CAM\imagenet-simple-labels.json'
image_file = r"PytorchAndPythonLearning\Pytorch\ModernCNN\CAM\test.jpg"

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 2
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features'  # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'

print(net)
for name, param in net.named_parameters():
    print(name, param.shape)

net.eval()

# hook the feature extractor
features_blobs = []


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())


def returnCAM(feature_conv: np.ndarray, weight_softmax: np.ndarray, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    """
    equals ::

    n_channels = 512
    output_tensor = t.zeros((512,49))
    for i in range(n_channels):
        output_tensor[i] = weight_softmax[i] * feature_conv [i,:].reshape(-1)
    output_tensor = output_tensor.sum(dim = 0)
    # output_tensor.shape is (49) , and it indicates how much attention should be paid to every location
    """
    for idx in class_idx:
        weight_softmax = weight_softmax[idx]
        feature_conv = feature_conv.reshape((nc, h*w))
        cam = weight_softmax.dot(feature_conv)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# load test image
img_pil = Image.open(image_file)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable)

# load the imagenet category list
with open(LABELS_file) as f:
    classes = json.load(f)


h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# render the CAM and output
print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
img = cv2.imread(image_file)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(
    CAMs[0], (width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite(
    r'.\PytorchAndPythonLearning\Pytorch\ModernCNN\CAM\CAM_result.jpg', result)
