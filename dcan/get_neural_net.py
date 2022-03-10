from torchvision import models


def get_alex_net():
    """
    Gets the original AlexNet.
    """
    alexnet = models.AlexNet

    return alexnet


def get_res_net():
    """
    Gets ResNet, which is more comples, and, possibly, better for our purposes.
    """
    resnet = models.ResNet

    return resnet
