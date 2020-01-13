import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import types
from collections import OrderedDict

from NAS_utils.ops import Conv2d_with_CD, Linear_with_CD

#Conv2d = Conv2d
Conv2d = Conv2d_with_CD
Linear = Linear_with_CD
nnConv2d = nn.Conv2d


__all__ = ['densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


input_sizes = {}
means = {}
stds = {}


for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]


pretrained_settings = {}


for model_name in __all__:
    pretrained_settings[model_name] = {
        'imagenet': {
            'url': model_urls[model_name],
            'input_space': 'RGB',
            'input_size': input_sizes[model_name],
            'input_range': [0, 1],
            'mean': means[model_name],
            'std': stds[model_name],
            'num_classes': 1000
        }
    }


def load_pretrained(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
    state_dict = model_zoo.load_url(settings['url'])
    state_dict = update_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']
    return model


def update_state_dict(state_dict):
    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, split=1, split_pattern=None):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False, split=split, split_pattern=split_pattern)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False, split=1)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        self._split_pattern = [num_input_features]
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, split=i+1, split_pattern=self._split_pattern)
            self.add_module('denselayer%d' % (i + 1), layer)
            self._split_pattern += [growth_rate]


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, split=1, split_pattern=None):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False, split=split, split_pattern=split_pattern))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, num_init_features//2, kernel_size=3, stride=2, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(num_init_features//2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(num_init_features//2, num_init_features//2, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(num_init_features//2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(num_init_features//2, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(num_init_features)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

            #('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,padding=3, bias=False)),
            #('norm0', nn.BatchNorm2d(num_init_features)),
            #('relu0', nn.ReLU(inplace=True)),
            #('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                    split=num_layers+1, split_pattern=[num_features - num_layers * growth_rate]+[growth_rate]*num_layers)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def _load_state_dict(model, model_url):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    state_dict = model_zoo.load_url(model_url)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def modify_densenets(model):
    # Modify attributs
    model.last_linear = model.classifier
    del model.classifier

    def logits(self, features):
        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def _densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls['densenet121'])
    return model

def _densenet169(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls['densenet169'])
    return model

def densenet121(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = _densenet121(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet121'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    #model = modify_densenets(model)
    return model


def densenet169(num_classes=1000, pretrained='imagenet'):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    model = _densenet169(pretrained=False)
    if pretrained is not None:
        settings = pretrained_settings['densenet169'][pretrained]
        model = load_pretrained(model, num_classes, settings)
    #model = modify_densenets(model)
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls['densenet201'])
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls['densenet161'])
    return model