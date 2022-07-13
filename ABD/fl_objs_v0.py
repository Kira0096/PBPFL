import torch
import torch.nn as nn

class Server(nn.Module):

    def __init__(self, model):
        super(Server, self).__init__()
        self.base_model = model
        self.current_model = self.base_model()

    def distribute(self):
    	distribute_model = self.base_model()
    	distribute_model.load_state_dict(self.current_model.state_dict())
    	return distribute_model

    def aggregate(self, client_model):
        self.counter += 1
        current_modules = self.current_model.fl_modules()
        
        for m_n, m in client_model.fl_modules().items():
            current_layer = current_modules[m_n]
            try:
                current_layer.set_grad((current_layer.get_grad() * (self.counter - 1) + m.get_grad()) / self.counter)
            except Exception as e:
                current_layer.set_grad(m.get_grad())
            

    def reset(self):
        self.counter = 0

    def update(self, lr):

        for m in self.current_model.fl_modules().items():
            m[1].update(float(lr) / self.counter)

            

class Client(nn.Module):

    def __init__(self, data_loader):
        super(Client, self).__init__()
        self.model = None
        self.loader = data_loader

    def receive_model(self, model):
        self.model = model

    def local_computation(self, y):
        self.model.post(y)
