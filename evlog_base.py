import json
import torch
from networks.main import build_network
from optim.trainer import EVLogTrainer


class EVLog(object):

    def __init__(self):
        self.uni_c = None
        self.local_c = None

    def set_network(self, meta_data):
        self.net = build_network(meta_data)

    def train(self, dataset, lr, n_epochs, lr_milestones, batch_size, weight_decay, device, optimizer_name='adam'):

        self.trainer = EVLogTrainer(self.uni_c, self.local_c, optimizer_name, lr=lr,
                                       n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                       weight_decay=weight_decay, device=device)
        self.net = self.trainer.train(dataset, self.net)
        # self.uni_c = self.trainer.uni_c.cpu().data.numpy().tolist()
        # self.local_c = self.trainer.local_c.cpu().data.numpy().tolist()

    def test(self, dataset):

        self.trainer.test(dataset, self.net)
