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
    	distribute_model.randomize()
    	return distribute_model

    def aggregate(self, client_model):
        self.counter += 1
        current_modules = self.current_model.fl_modules()
        
        for m_n, m in client_model.fl_modules().items():
            current_layer = current_modules[m_n]
            current_layer.aggregate_grad(current_layer.correction(client_model.gamma, client_model.v, m.post_data, m.get_grad(), m.get_r()), self.counter)

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
