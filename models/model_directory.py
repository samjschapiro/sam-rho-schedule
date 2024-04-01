import torchvision.models as models
from torch import nn
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_resnet18(num_classes=10, pretrained=False):
    if pretrained:
        resnet = models.resnet18(weights="DEFAULT").to(device)
    else:
        resnet = models.resnet18(weights=None).to(device)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes).to(device)
    return resnet


def get_resnet34(num_classes=10, pretrained=False):
    if pretrained:
        resnet = models.resnet34(weights="DEFAULT").to(device)
    else:
        resnet = models.resnet34(weights=None).to(device)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes).to(device)
    return resnet


def get_resnet50(num_classes=10, pretrained=False):
    if pretrained:
        resnet = models.resnet50(weights="DEFAULT")
    else:
        resnet = models.resnet50(weights=None)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet

def get_resnet101(num_classes=10, pretrained=False):
    if pretrained:
        resnet = models.resnet101(weights="DEFAULT").to(device)
    else:
        resnet = models.resnet101(weights=None).to(device)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes).to(device)
    return resnet


def get_vgg11(num_classes=10):
    return models.vgg11(num_classes=num_classes).to(device)


def get_vgg13(num_classes=10):
    return models.vgg13(num_classes=num_classes).to(device)


def get_vgg16(num_classes=10):
    return models.vgg16_bn(num_classes=num_classes).to(device)


def get_vgg19(num_classes=10):
    return models.vgg19(num_classes=num_classes).to(device)


def get_wide_resnet50_2(pretrained=False, **kwargs):
    return models.wide_resnet50_2(pretrained=pretrained, **kwargs).to(device)


def get_wide_resnet101_2(pretrained=False, **kwargs):
    return models.wide_resnet101_2(pretrained=pretrained, **kwargs).to(device)


def get_densenet121(pretrained=False, **kwargs):
    return models.densenet121(pretrained, **kwargs).to(device)


def get_densenet161(pretrained=False, **kwargs):
    return models.densenet161(pretrained, **kwargs).to(device)


def get_densenet201(pretrained=False, **kwargs):
    return models.densenet201(pretrained, **kwargs).to(device)


MODEL_GETTERS = {
    "resnet18": get_resnet18,
    "resnet34": get_resnet34,
    "resnet50": get_resnet50,
    "resnet101": get_resnet101,
    "vgg11": get_vgg11,
    "vgg13": get_vgg13,
    "vgg16": get_vgg16,
    "vgg19": get_vgg19,
    "wide_resnet50_2": get_wide_resnet50_2,
    "wide_resnet101_2": get_wide_resnet101_2,
    "densenet121": get_densenet121,
    "densenet161": get_densenet161,
    "densenet201": get_densenet201,
}
