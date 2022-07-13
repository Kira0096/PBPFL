
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

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

    def __init__(self, num_in, num_out, last=False):
        super(Linear, self).__init__()
        self.fc = nn.Linear(num_in, num_out, bias=False)
        self.num_in, self.num_out = num_in, num_out
        self.last = last
    def forward(self, x):
        self.y = self.fc(x)
        return self.y

    def randomize(self, last_r = None, set_r = None):

        if set_r is None:
            self.rc = torch.ones(self.fc.weight.size(0)).to(0) + 1e-5
        else:
            self.rc = set_r
        
        if last_r is None:
            self.r = torch.einsum('i,j->ij', self.rc, torch.ones(self.fc.weight.size(1)).to(0))
        else:
            self.r = torch.einsum('i,j->ij', self.rc, 1. / last_r.repeat(int(self.num_in / last_r.size(0)), 1).t().reshape(-1))

        if self.last:
            self.a, self.b, self.gamma = torch.rand(self.num_out).to(0) + 1e-5, torch.rand(self.num_out).to(0) + 1e-5, torch.rand(2).to(0)  + 1e-5
            self.gamma[1] = 0
            self.rcls = self.a * self.gamma[0] + self.b * self.gamma[1]

            self.v = torch.sum(self.rcls ** 2)

            self.fc.weight = torch.nn.Parameter(self.r * self.fc.weight + self.rcls.view(-1, 1))

            return self.a, self.b, self.gamma, self.v
        else:
            self.fc.weight = torch.nn.Parameter(self.r * self.fc.weight)            
            return self.rc

    def post(self, y_logit, y_gt, y_last, global_var):

        if self.last:
            a, b, alpha, cls_weight = global_var
            batch_size = y_last.size(0)
            delta_y = (y_logit - y_gt).view(batch_size, -1)
            y_last = y_last.view(batch_size, -1)
            sigma_conv = torch.einsum('kj,k->kj', y_last, alpha)
            sigma_conv_a = torch.einsum('i,kj->kij', a, sigma_conv).mean(dim=0)
            sigma_conv_b = torch.einsum('i,kj->kij', b, sigma_conv).mean(dim=0)
            self.post_data = sigma_conv_a, sigma_conv_b, 0
        else:
            self.post_data = _post(y_logit, y_gt, y_last, global_var, self.y, self.fc.weight)
        
    def correction(self, gamma, v, post_data, grad, r):

        if self.last:
            sigma_conv_a, sigma_conv_b, _ = post_data
            sigma_conv = gamma[0] * sigma_conv_a + gamma[1] * sigma_conv_b
            
            delta_conv = sigma_conv

        else:
            sigma_conv_a, sigma_conv_b, beta = post_data
            sigma_conv = gamma[0] * sigma_conv_a + gamma[1] * sigma_conv_b
            beta = beta * v
            delta_conv = sigma_conv - beta

        return (grad - delta_conv) * r
        

    def get_grad(self):
        return self.fc.weight.grad
    def set_grad(self, grad):
        self.fc.weight.grad = grad

    def get_r(self):
        return self.r

    def aggregate_grad(self, grad, counter):
        try:
            self.fc.weight.grad = (self.fc.weight.grad * (counter - 1) + grad) / counter
        except Exception as e:
            self.fc.weight.grad = grad

    def update(self, lr):
        self.fc.weight = nn.Parameter(self.fc.weight - self.fc.weight.grad * lr)

        


class MLP(nn.Module):

    def __init__(self, num_in, num_out, num_layers=3):
        super(MLP, self).__init__()
        self.fc1 = Linear(num_in, 50)
        self.fc2 = Linear(50, 50)
        middle = []
        for i in range((num_layers - 3)):
            middle.append(Linear(50, 50))
        self.middle = nn.Sequential(*middle)
        self.fc3 = Linear(50, 50)
        self.last = Linear(50, num_out, True)

    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        for i in range(len(self.middle)):
            y2 = F.relu(self.middle[i](y2))
        y3 = F.relu(self.fc3(y2))
        self.y_last = y3
        self.logits = self.last(y3)
        return self.logits

    def randomize(self):
        r1 = self.fc1.randomize()
        r2 = self.fc2.randomize(r1)
        for i in range(len(self.middle)):
            r2 = self.middle[i].randomize(r2)
        r3 = self.fc3.randomize(r2)
        self.a, self.b, self.gamma, self.v = self.last.randomize(r3)

    def post(self, y_gt):
        batch_size = y_gt.size(0)

        delta_y = self.logits - y_gt

        self.alpha = self.y_last.view(batch_size, -1).sum(dim=-1)
        global_var = self.a, self.b, self.alpha, None

        opt = torch.optim.SGD(self.parameters(), 0)

        logits = self.logits.view(batch_size, -1)
        y_last = self.y_last.view(batch_size, -1)
        delta_y = delta_y.view(batch_size, -1)

        opt.zero_grad()
        
        logits.backward(torch.einsum('c,k->kc', self.a, self.alpha), retain_graph=True)
        for m in self.named_modules():
            if type(m[1]) is Linear:
                m[1].sigma_conv_a = m[1].fc.weight.grad / batch_size
                m[1].sigma_conv_b = 0

        opt.zero_grad()
        y_last.backward(torch.einsum('k,kl->kl', torch.einsum('kc,c->k', delta_y, self.a), torch.ones_like(y_last)), retain_graph=True)
        for m in self.named_modules():
            if type(m[1]) is Linear:
                m[1].sigma_conv_a += (m[1].fc.weight.grad / batch_size)

        opt.zero_grad()
        y_last.backward(torch.einsum('k,kl->kl', self.alpha, torch.ones_like(y_last)), retain_graph=True)
        for m in self.named_modules():
            if type(m[1]) is Linear:
                m[1].beta = m[1].fc.weight.grad / batch_size
                m[1].post_data = (m[1].sigma_conv_a, 0, m[1].beta)
        opt.zero_grad()

    def correction(self):
        self.fc1.correction(self.gamma, self.v)
        self.fc2.correction(self.gamma, self.v)
        self.fc3.correction(self.gamma, self.v)
        self.last.correction(self.gamma, self.v)

    def update(self,lr):

        for idx, m in enumerate(self.children()):
            m.update(lr)

    def tell(self):

        for idx, m in enumerate(self.named_children()):
            print (idx, '-', m)
            # m.update(0)
            print (getattr(self, m[0]).update(0))

    def fl_modules(self):
        module_dict = {}
        for idx, m in enumerate(self.named_modules()):
            if type(m[1]) is Linear:
                module_dict[m[0]] = m[1]
        return module_dict

if __name__ == '__main__':
    import numpy as np
    def criterion(y_pred, y_cls):
        return ((y_pred - y_cls)**2).sum() / 2.

    batch_size = 30
    net = MLP(5, 1).to(0)

    opt = torch.optim.SGD(net.parameters(), 0.0)

    x = torch.rand((batch_size, 5)).to(0)
    y_gt = torch.rand((batch_size, 1)).to(0) * 100

    y_pred = net(x)
    
    loss = criterion(y_pred, y_gt) 
    loss.backward()
    ag1 = net.fc1.fc.weight.grad.detach().cpu().numpy().copy()

    opt.zero_grad()

    net.randomize()

    y_pred_r = net(x)
    net.post(y_gt)
    loss = criterion(y_pred_r, y_gt) 
    loss.backward()

    net.correction()
    ag2 = net.fc1.fc.weight.grad.detach().cpu().numpy().copy()

    print (ag1.reshape(-1)[:10], ag2.reshape(-1)[:10])
    print (np.abs((ag1 - ag2)).mean())
    