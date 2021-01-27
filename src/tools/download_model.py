from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.xception import Xception


def download():

    res_net = ResNet50(weights="imagenet")
    mobile_net = MobileNet(weights="imagenet")
    x_ception = Xception(weights="imagenet")

    return [res_net, mobile_net, x_ception]
