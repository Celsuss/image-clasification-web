from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.xception import Xception


def download(model_dict={}):

    res_net = ResNet50(weights="imagenet")
    model_dict["resnet50"] = res_net

    mobile_net = MobileNet(weights="imagenet")
    model_dict["mobilenet"] = mobile_net

    x_ception = Xception(weights="imagenet")
    model_dict["xception"] = x_ception

    return [res_net, mobile_net, x_ception]
