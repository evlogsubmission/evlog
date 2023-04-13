from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import logging
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import collections
from tqdm import tqdm


class EVLogTrainer():

    def __init__(self, uni_c, local_c, optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device):

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.uni_c = torch.tensor(uni_c, device=self.device) if uni_c is not None else None
        self.local_c = torch.tensor(local_c, device=self.device) if local_c is not None else None



    def train(self, dataset, net):
        logger = logging.getLogger()

        net = net.to(self.device)

        # Get train data loader
        train_loader, _, _ = dataset.loaders(batch_size=self.batch_size)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.uni_c is None or self.local_c is None:
            logger.info('Initializing center c...')
            self.uni_c, self.local_c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _, local_feat, global_feat = data
                inputs = inputs.to(self.device)
                local_feat = local_feat.to(self.device)
                global_feat = global_feat.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                uni_output, local_output, global_output = net(inputs, local_feat, global_feat)
                uni_dist = torch.sum((uni_output - self.uni_c) ** 2, dim=1)
                local_dist = torch.sum((local_output - self.local_c) ** 2, dim=1)

                loss = torch.mean(uni_dist) + torch.mean(local_dist)
                    # loss = torch.mean(uni_dist)
                    # regx = torch.min(dist)
                    # loss = radiu_loss - 0.0001 * regx

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, dataset, net):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        train_loader, test_loader, trans_loader = dataset.loaders(batch_size=self.batch_size)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()

        dists = torch.tensor([0]).float()
        with torch.no_grad():
            for data in train_loader:
                inputs, labels, idx, local_feat, global_feat = data
                inputs = inputs.to(self.device)
                local_feat = local_feat.to(self.device)
                global_feat = global_feat.to(self.device)
                uni_output, local_output, global_output = net(inputs, local_feat, global_feat)
                uni_dist = torch.sum((uni_output - self.uni_c) ** 2, dim=1)
                local_dist = torch.sum((local_output - self.local_c) ** 2, dim=1)
                scores = uni_dist + local_dist
                dists = torch.cat((dists, scores.cpu().squeeze()), dim=0)
        # max
        radius = torch.max(dists).item()


        print('Max distance in training set:', radius)


        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx, local_feat, global_feat = data
                inputs = inputs.to(self.device)
                local_feat = local_feat.to(self.device)
                global_feat = global_feat.to(self.device)
                uni_output, local_output, global_output = net(inputs, local_feat, global_feat)
                uni_dist = torch.sum((uni_output - self.uni_c) ** 2, dim=1)
                local_dist = torch.sum((local_output - self.local_c) ** 2, dim=1)

                scores = uni_dist + local_dist
                    #  scores = uni_dist

                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        idxs, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))


        # visualize wrong
        # for cur_radius in np.arange(-0.0005,0.0005,0.0001):
        for radiu in np.arange(0.1, 1.5, 0.1):
            wrong_set = list()
            wrong_dist = dict()
        #
            cur_radius = radiu * radius
            preds = (scores > cur_radius).astype(int)
            for idx, label, pred, score in zip(idxs, labels, preds, scores):
                if label != pred:
                    wrong_set.append(dataset.test_set.retrieve(idx)[0])
                    wrong_dist[dataset.test_set.retrieve(idx)[0]] = (score, label)
                #     print(dataset.test_set.retrieve(idx))

            # print('====== Current radius:', radiu * radius, '|ratio:', radiu, '======')
            # print('====== Current radius:', cur_radius, '======')
            # # print(collections.Counter(wrong_set))

            eval_results = {
                "f1": f1_score(labels, preds),
                "rc": recall_score(labels, preds),
                "pc": precision_score(labels, preds),
                "acc": accuracy_score(labels, preds),
            }
            logging.info('Ratio: {} | Current radius: {} |'.format(radiu, cur_radius) + str({k: f"{v:.3f}" for k, v in eval_results.items()}))
            counter_wrong_set = collections.Counter(wrong_set)
            wrong_info = {k:(v[0], v[1], counter_wrong_set[k]) for k, v in wrong_dist.items()}
        logger.info('End testing, start evaluate evolving...')


        with torch.no_grad():
            for data in trans_loader:
                inputs, labels, idx, local_feat, global_feat = data
                inputs = inputs.to(self.device)
                local_feat = local_feat.to(self.device)
                global_feat = global_feat.to(self.device)
                uni_output, local_output, global_output = net(inputs, local_feat, global_feat)
                uni_dist = torch.sum((uni_output - self.uni_c) ** 2, dim=1)
                local_dist = torch.sum((local_output - self.local_c) ** 2, dim=1)
                scores = uni_dist + local_dist
                    # scores = uni_dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        idxs, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))


        # visualize wrong
        for radiu in np.arange(0.1, 1.5, 0.1):
            cur_radius = radiu * radius
            preds = (scores > cur_radius).astype(int)
            eval_results = {
                "f1": f1_score(labels, preds),
                "rc": recall_score(labels, preds),
                "pc": precision_score(labels, preds),
                "acc": accuracy_score(labels, preds),
            }
            logging.info('Ratio: {} | Current radius: {} |'.format(radiu, cur_radius) + str({k: f"{v:.3f}" for k, v in eval_results.items()}))

        logger.info('Finished.')

    def init_center_c(self, train_loader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        uni_c = torch.zeros(net.rep_dim, device=self.device)
        local_c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in tqdm(train_loader):
                # get the inputs of the batch
                inputs, _, _, local_feat, global_feat = data
                inputs = inputs.to(self.device)
                local_feat = local_feat.to(self.device)
                global_feat = global_feat.to(self.device)
                uni_output, local_output, global_output = net(inputs, local_feat, global_feat)
                n_samples += uni_output.shape[0]
                uni_c += torch.sum(uni_output, dim=0)
                local_c += torch.sum(local_output, dim=0)

        uni_c /= n_samples
        local_c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        uni_c[(abs(uni_c) < eps) & (uni_c < 0)] = -eps
        uni_c[(abs(uni_c) < eps) & (uni_c > 0)] = eps

        local_c[(abs(local_c) < eps) & (local_c < 0)] = -eps
        local_c[(abs(local_c) < eps) & (local_c > 0)] = eps

        return uni_c, local_c

