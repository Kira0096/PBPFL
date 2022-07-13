
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

def _post(y_logit, y_gt, y_last, global_var, y_current, weight):
    batch_size = y_last.size(0)

    a, b, alpha, cls_weight = global_var

    y_last = y_last.view(batch_size, -1)
    _y_cur = y_current.view(batch_size, -1)
    delta_y = y_logit - y_gt
    delta_y = delta_y.view(batch_size, -1)
    y_logit, y_gt = y_logit.view(batch_size, -1), y_gt.view(batch_size, -1)
    
    sigma_conv_a = torch.autograd.grad(y_logit, weight, torch.einsum('c,k->kc', a, alpha), retain_graph=True)[0]
    sigma_conv_a += torch.autograd.grad(y_last, weight, torch.einsum('k,kl->kl', torch.einsum('kc,c->k', delta_y, a), torch.ones_like(y_last)), retain_graph=True)[0]

    sigma_conv_b = torch.autograd.grad(y_logit, weight, torch.einsum('c,k->kc', b, alpha), retain_graph=True)[0]
    sigma_conv_b += torch.autograd.grad(y_last, weight, torch.einsum('k,kl->kl', torch.einsum('kc,c->k', delta_y, b), torch.ones_like(y_last)), retain_graph=True)[0]

    beta = torch.autograd.grad(y_last, weight, torch.einsum('k,kl->kl', alpha, torch.ones_like(y_last)), retain_graph=True)[0]

    return sigma_conv_a / batch_size, sigma_conv_b / batch_size, beta / batch_size

class Linear(nn.Module):

    def __init__(self, num_in, num_out):
        super(Linear, self).__init__()
        self.fc = nn.Linear(num_in, num_out, bias=False)
        self.num_in, self.num_out = num_in, num_out
    def forward(self, x):
        self.y = self.fc(x)
        return self.y

    def randomize(self, last_r = None, set_r = None):

        if set_r is None:
            self.rc1 = torch.ones(self.fc.weight.size(0)).to(0) + 1e-5
        else:
            self.rc1 = set_r
        
        if last_r is None:
            self.r1 = torch.einsum('i,j->ij', self.rc1, torch.ones(self.fc.weight.size(1)).to(0))
        else:
            self.r1 = torch.einsum('i,j->ij', self.rc1, 1. / last_r.repeat(int(self.num_in / last_r.size(0)), 1).t().reshape(-1))

        self.a, self.b, self.gamma = torch.rand(self.num_out).to(0) + 1e-5, torch.rand(self.num_out).to(0) + 1e-5, torch.rand(2).to(0)  + 1e-5
        self.gamma[1] = 0
        self.rcls = self.a * self.gamma[0] + self.b * self.gamma[1]
        self.v = torch.sum(self.rcls ** 2)

        self.fc.weight = torch.nn.Parameter(self.r1 * self.fc.weight + self.rcls.view(-1, 1))

        return self.a, self.b, self.gamma, self.v

    def post(self, y_logit, y_gt, y_last, global_var):

        a, b, alpha, cls_weight = global_var
        batch_size = y_last.size(0)
        delta_y = (y_logit - y_gt).view(batch_size, -1)
        y_last = y_last.view(batch_size, -1)
        sigma_conv = torch.einsum('kj,k->kj', y_last, alpha)
        sigma_conv_a = torch.einsum('i,kj->kij', a, sigma_conv).sum(dim=0)
        sigma_conv_b = torch.einsum('i,kj->kij', b, sigma_conv).sum(dim=0)

        self.post_data = sigma_conv_a / batch_size, sigma_conv_b / batch_size
        

    def correction(self, gamma, v, post_data, grad, r):

        sigma_conv_a, sigma_conv_b = post_data
        sigma_conv = gamma[0] * sigma_conv_a + gamma[1] * sigma_conv_b
        
        delta_conv1 = sigma_conv

        return (grad - delta_conv1) * r

    def get_grad(self):
        return self.fc.weight.grad
    def set_grad(self, grad):
        self.fc.weight.grad = grad

    def get_r(self):
        return self.r1

    def aggregate_grad(self, grad, counter):
        try:
            self.fc.weight.grad = (self.fc.weight.grad * (counter - 1) + grad) / counter
        except Exception as e:
            self.fc.weight.grad = grad

    def update(self, lr):
        self.fc.weight = nn.Parameter(self.fc.weight - self.fc.weight.grad * lr)

class Conv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.in_channel, self.out_channel = in_channel, out_channel

    def forward(self, x):
        self.y = self.conv(x)
        return self.y

    def randomize(self, last_r = None, set_r = None):
        if set_r is None:
            self.rc = torch.rand(self.out_channel).to(0) + 1e-5
        else:
            self.rc = set_r
        
        if last_r is None:
            self.r = torch.einsum('i,j->ij', self.rc, torch.ones(self.in_channel).to(0))
        else:
            self.r = torch.einsum('i,j->ij', self.rc, 1. / last_r)
        
        self.conv.weight = torch.nn.Parameter(self.conv.weight * self.r.unsqueeze(-1).unsqueeze(-1))
        
        return self.rc

    def post(self, y_logit, y_gt, y_last, global_var):

        self.post_data = _post(y_logit, y_gt, y_last, global_var, self.y, self.conv.weight)
        

    def correction(self, gamma, v, post_data, grad, r):

        sigma_conv_a, sigma_conv_b, beta = post_data
        sigma_conv = gamma[0] * sigma_conv_a + gamma[1] * sigma_conv_b
        beta = beta * v
        delta_conv1 = sigma_conv - beta

        return (grad - delta_conv1) * r.unsqueeze(-1).unsqueeze(-1)


    def get_grad(self):
        return self.conv.weight.grad
    def set_grad(self, grad):
        self.conv.weight.grad = grad

    def get_r(self):
        return self.r

    def aggregate_grad(self, grad, counter):
        try:
            self.conv.weight.grad = (self.conv.weight.grad * (counter - 1) + grad) / counter
        except Exception as e:
            self.conv.weight.grad = grad

    def update(self, lr):
        self.conv.weight = nn.Parameter(self.conv.weight - self.conv.weight.grad * lr)

        

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.Sequential()
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def randomize(self, last_r):
        self.r1 = self.conv1.randomize(last_r)
        
        if len(self.shortcut) > 0:
            self.r2 = self.conv2.randomize(self.r1)
            self.shortcut[0].randomize(last_r, self.r2)
        else:
            self.r2 = self.conv2.randomize(self.r1, last_r)
        return self.r2

    def post(self, y_logit, y_gt, y_last, global_var):
        self.conv1.post(y_logit, y_gt, y_last, global_var)
        self.conv2.post(y_logit, y_gt, y_last, global_var)
        if len(self.shortcut) > 0:
            self.shortcut[0].post(y_logit, y_gt, y_last, global_var)

    def correction(self, gamma, v):
        self.conv1.correction(gamma, v)
        self.conv2.correction(gamma, v)
        if len(self.shortcut) > 0:
            self.shortcut[0].correction(gamma, v)



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.Sequential()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        self.y_last = out
        out = self.linear(out)
        self.logits = out
        return out

    def randomize(self):
        self.r1 = self.conv1.randomize()
        r = self.r1
        
        for i in range(len(self.layer1)):
            r = self.layer1[i].randomize(r)
        self.r2 = r
        for i in range(len(self.layer2)):
            r = self.layer2[i].randomize(r)
        self.r3 = r
        for i in range(len(self.layer3)):
            r = self.layer3[i].randomize(r)
        self.r4 = r

        self.a, self.b, self.gamma, self.v = self.linear.randomize(self.r4, torch.ones(self.linear.fc.weight.size(0)).to(0))

    def post(self, y_gt):
        batch_size = y_gt.size(0)

        delta_y = self.logits - y_gt
        cls_weight = self.linear.fc.weight
        self.alpha = self.y_last.view(batch_size, -1).sum(dim=-1)

        global_var = (self.a, self.b, self.alpha, cls_weight)
        opt = torch.optim.SGD(self.parameters(), 0)


        logits = self.logits.view(batch_size, -1)
        y_last = self.y_last.view(batch_size, -1)
        delta_y = delta_y.view(batch_size, -1)

        opt.zero_grad()
        
        logits.backward(torch.einsum('c,k->kc', self.a, self.alpha), retain_graph=True)
        for m in self.named_modules():
            if type(m[1]) is Conv2d:
                m[1].sigma_conv_a = m[1].conv.weight.grad / batch_size
                m[1].sigma_conv_b = 0

        opt.zero_grad()
        y_last.backward(torch.einsum('k,kl->kl', torch.einsum('kc,c->k', delta_y, self.a), torch.ones_like(y_last)), retain_graph=True)
        for m in self.named_modules():
            if type(m[1]) is Conv2d:
                m[1].sigma_conv_a += (m[1].conv.weight.grad / batch_size)

        opt.zero_grad()
        y_last.backward(torch.einsum('k,kl->kl', self.alpha, torch.ones_like(y_last)), retain_graph=True)
        for m in self.named_modules():
            if type(m[1]) is Conv2d:
                m[1].beta = m[1].conv.weight.grad / batch_size
                m[1].post_data = (m[1].sigma_conv_a, 0, m[1].beta)
        opt.zero_grad()

        self.linear.post(self.logits, y_gt, self.y_last, global_var)

    def correction(self):

        self.conv1.correction(self.gamma, self.v)
        for i in range(len(self.layer1)):
            self.layer1[i].correction(self.gamma, self.v)
        for i in range(len(self.layer2)):
            self.layer2[i].correction(self.gamma, self.v)
        for i in range(len(self.layer3)):
            self.layer3[i].correction(self.gamma, self.v)
        self.linear.correction(self.gamma, self.v)


    def fl_modules(self):
        module_dict = {}
        for idx, m in enumerate(self.named_modules()):
            if type(m[1]) is Conv2d or type(m[1]) is Linear:
                
                module_dict[m[0]] = m[1]
        return module_dict




def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
