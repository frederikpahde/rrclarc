import torch
import torch.hub
from torch import Tensor
from torchvision.models import resnet18, resnet34, resnet50

from utils.lrp_canonizers import ResNetCanonizer


def get_resnet18(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_resnet(resnet18, ckpt_path, pretrained, n_class)


def get_resnet34(ckpt_path=None, pretrained=True, n_class=None) -> torch.nn.Module:
    return get_resnet(resnet34, ckpt_path, pretrained, n_class)


def get_resnet50(ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    return get_resnet(resnet50, ckpt_path, pretrained, n_class)


def get_resnet(model_fn, ckpt_path=None, pretrained=True, n_class: int = None) -> torch.nn.Module:
    if pretrained:
        weights = "IMAGENET1K_V1"
    else:
        weights = None

    model = model_fn(weights=weights)

    if n_class and n_class != 1000:
        num_in = model.fc.in_features
        model.fc = torch.nn.Linear(num_in, n_class, bias=True)
    if ckpt_path:
        checkpoint = torch.load(ckpt_path)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if "module" in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

    model.identity_0 = torch.nn.Identity()
    model.relu_0 = torch.nn.ReLU(inplace=False)
    model.identity_1 = torch.nn.Identity()
    model.relu_1 = torch.nn.ReLU(inplace=False)
    model.identity_2 = torch.nn.Identity()
    model.relu_2 = torch.nn.ReLU(inplace=False)
    model.last_conv = torch.nn.Identity()
    model.last_relu = torch.nn.ReLU(inplace=False)
    model._forward_impl = _forward_impl_.__get__(model)
    model.layer4[-1].last_conv = torch.nn.Identity()
    model.layer4[-1].forward = forward_BB.__get__(model.layer4[-1])

    return model


def _forward_impl_(self, x):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.identity_0(x)  # added identity
    x = self.relu_0(x)

    x = self.layer2(x)
    x = self.identity_1(x)  # added identity
    x = self.relu_1(x)

    x = self.layer3(x)
    x = self.identity_2(x)  # added identity
    x = self.relu_2(x)

    x = self.layer4(x)
    x = self.last_relu(self.last_conv(x))  # added identity

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x


def forward_BB(self, x: Tensor) -> Tensor:
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.last_conv(out)
    out = out + 0
    out = self.relu(out)

    return out


def get_resnet_canonizer():
    return ResNetCanonizer()
