import numpy as np

import torch as t
import torch.nn.functional as F

from torch.nn import Module, Conv2d, Linear

TRAINING_SIZE = 1281167.0

class Conv2d_with_CD(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 weight_reg=10.0, drop_reg=1.0, p_init=1e-1, deterministic=True, debug=False, split=1, split_pattern=None):
        super(Conv2d_with_CD, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups, bias=bias)

        self._weight_reg = weight_reg / TRAINING_SIZE
        self._drop_reg = drop_reg / TRAINING_SIZE
        self._det_mode = deterministic
        self._debug_mode = debug
        self._deterministic = deterministic

        self.split = split
        self.split_pattern = split_pattern
        if self.split_pattern:
            assert len(self.split_pattern) == self.split
            assert sum(split_pattern) == self.in_channels

        self._noise_shape = (self.in_channels, 1, 1)
        self._eps = 1e-8
        self._temp = 1. / 5.
        self._p_init = p_init

        self.p_logit = t.nn.Parameter(t.Tensor([np.log(self._p_init) - np.log(1. - self._p_init)]*self.split))

        if self._deterministic:
            print('Using determinist drop.')
            self.unif_noise_var = t.zeros(size=[1]+list(self._noise_shape))#.uniform_(0,1)
            self.unif_noise_variable = t.nn.Parameter(self.unif_noise_var, requires_grad=False)

        if self._debug_mode:
            if self.in_channels == 64:
                self.p_logit.register_hook(print)

    def _concrete_dropout(self, input):
        if self.split_pattern:
            _p = self.p_logit[0].expand(self.split_pattern[0])
            if self.split > 1:
                _p = t.cat(
                    (_p, self.p_logit[1:].view(-1,1).expand(self.split-1, self.split_pattern[1]).reshape(-1)),
                    dim=0
                )
        else:
            assert self.split == 1
            _p = self.p_logit.expand(self.in_channels)
        _p = _p.sigmoid().view([1]+list(self._noise_shape))

        if self._deterministic:
            drop_tensor = t.floor(self.unif_noise_variable.cuda() + _p)
            random_tensor = 1. - drop_tensor
        else:
            unif_noise_1 = t.rand(size=[input.shape[0]]+list(self._noise_shape)).cuda()
            unif_noise_2 = t.rand(size=[input.shape[0]]+list(self._noise_shape)).cuda()

            drop_prob = (
                t.log(_p + self._eps)
                - t.log(1. - _p + self._eps)
                + t.log(-t.log(unif_noise_1 + self._eps) + self._eps)
                - t.log(-t.log(unif_noise_2 + self._eps) + self._eps)
            )

            drop_prob = t.sigmoid(drop_prob/self._temp)
            random_tensor = 1. - drop_prob

        return input * random_tensor


    def forward(self, input):
        input = self._concrete_dropout(input)
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    @property
    def KLreg(self):
        if self.split_pattern:
            _p = self.p_logit[0].expand(self.split_pattern[0])
            if self.split > 1:
                _p = t.cat(
                    (_p, self.p_logit[1:].view(-1,1).expand(self.split-1, self.split_pattern[1]).reshape(-1)),
                    dim=0
                )
        else:
            assert self.split == 1
            _p = self.p_logit[0].expand(self.in_channels)
        _p = _p.sigmoid()
        #weight_regularizer = self._weight_reg * t.sum(self.weight**2) * (1. - _p)
        # deprecated by split version
        weight_regularizer = self._weight_reg * t.sum((self.weight**2)* (1. - _p.view([1, self.in_channels, 1, 1])))
        dropout_regularizer = _p * t.log(_p)
        dropout_regularizer += (1. - _p) * t.log(1. - _p)
        #dropout_regularizer *= self._drop_reg * self.in_channels
        # deprecated by split version
        dropout_regularizer *= self._drop_reg
        return weight_regularizer + t.sum(dropout_regularizer)

    @property
    def p(self):
        return self.p_logit.sigmoid()


class Linear_with_CD(Linear):
    def __init__(self, in_features, out_features, bias=True,
                 weight_reg=10.0, drop_reg=1.0, p_init=1e-1, deterministic=False, debug=False, split=1, split_pattern=None):
        super(Linear_with_CD, self).__init__(in_features, out_features, bias)

        self._weight_reg = weight_reg / TRAINING_SIZE
        self._drop_reg = drop_reg / TRAINING_SIZE
        self._det_mode = deterministic
        self._debug_mode = debug

        self.split = split
        self.split_pattern = split_pattern
        if self.split_pattern:
            assert len(self.split_pattern) == self.split
            assert sum(split_pattern) == self.in_features

        self._noise_shape = (self.in_features,)
        self._eps = 1e-8
        self._temp = 1. / 5.
        self._p_init = p_init

        if split_pattern:
            self.p_logit = t.Tensor([np.log(self._p_init) - np.log(1. - self._p_init)]).expand(self.split_pattern[0])
            for sidx in range(1, self.split):
                self.p_logit = t.cat(
                    (self.p_logit, t.Tensor([np.log(self._p_init) - np.log(1. - self._p_init)]).expand(self.split_pattern[sidx])),
                    dim=0
                )
        else:
            assert self.split == 1
            self.p_logit = t.Tensor([np.log(self._p_init) - np.log(1. - self._p_init)]).expand(self.in_channels)
        self.p_logit = t.nn.Parameter(self.p_logit)

        if self._debug_mode:
            self.p_logit.register_hook(print)

    def _concrete_dropout(self, input):
        _p = self.p_logit.sigmoid().view([1] + list(self._noise_shape))
        unif_noise_1 = t.rand(size=[input.shape[0]]+list(self._noise_shape)).cuda()
        unif_noise_2 = t.rand(size=[input.shape[0]]+list(self._noise_shape)).cuda()

        drop_prob = (
            t.log(_p + self._eps)
            - t.log(1. - _p + self._eps)
            + t.log(-t.log(unif_noise_1 + self._eps) + self._eps)
            - t.log(-t.log(unif_noise_2 + self._eps) + self._eps)
        )

        drop_prob = t.sigmoid(drop_prob/self._temp)
        random_tensor = 1. - drop_prob

        return input * random_tensor

    def forward(self, input):
        input = self._concrete_dropout(input)
        return F.linear(input, self.weight, self.bias)

    @property
    def KLreg(self):
        _p = self.p_logit.sigmoid()
        #weight_regularizer = self._weight_reg * t.sum(self.weight ** 2) * (1. - _p)
        # deprecated by split version
        weight_regularizer = self._weight_reg * t.sum((self.weight ** 2) * (1. - _p.view([1, self.in_features, 1, 1])))
        dropout_regularizer = _p * t.log(_p)
        dropout_regularizer += (1. - _p) * t.log(1. - _p)
        #dropout_regularizer *= self._drop_reg * self.in_features
        # deprecated by split version
        dropout_regularizer *= self._drop_reg
        return t.sum(weight_regularizer + dropout_regularizer)

    @property
    def p(self):
        return self.p_logit.sigmoid()