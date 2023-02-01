import torchvision
import torch.nn as nn


def ResNet18_Clothing(pretrained=True, classes=14):
    model = torchvision.models.resnet18(pretrained=pretrained, num_classes=1000)
    model.fc = nn.Linear(512, classes, bias=True)
    return model


def ResNet34_Clothing(pretrained=True, classes=14):
    model = torchvision.models.resnet18(pretrained=pretrained, num_classes=1000)
    model.fc = nn.Linear(512, classes, bias=True)
    return model


def ResNet50_Clothing(pretrained=True, classes=14):
    model = torchvision.models.resnet50(pretrained=pretrained, num_classes=1000)
    model.fc = nn.Linear(2048, classes, bias=True)
    return model


def densenet121_Clothing(pretrained=True, classes=14):
    model = torchvision.models.densenet121(pretrained=pretrained, num_classes=1000)
    model.classifier = nn.Linear(1024, classes, bias=True)
    return model


def mobilenet_v2_Clothing(pretrained=True, classes=14):
    model = torchvision.models.mobilenet_v2(pretrained=pretrained, num_classes=1000)
    model.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                     nn.Linear(model.last_channel, classes), )
    return model


def inception_v3_Clothing(pretrained=True, classes=14):
    model = torchvision.models.inception_v3(pretrained=pretrained, num_classes=1000, aux_logits=True)
    model.fc = nn.Linear(2048, classes, bias=True)
    return model


def googlenet_Clothing(pretrained=True, classes=14):
    model = torchvision.models.googlenet(pretrained=pretrained, num_classes=1000, aux_logits=True)
    model.fc = nn.Linear(1024, classes, bias=True)
    return model
