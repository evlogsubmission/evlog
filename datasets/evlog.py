from torch.utils.data import Subset
import pickle

from torch.utils.data import Dataset
import itertools
import torch
import os
import json
from collections import Counter, defaultdict
import re
import logging
from tqdm import tqdm
import numpy as np

from torch.utils.data import DataLoader
from simcse import SimCSE

import io
import hdbscan
from sklearn.externals import joblib
from sklearn.decomposition import PCA


class SemanticCluster():

    def __init__(self, cluster_epsilon=0.1, n_components=50):

        self.clusterer = hdbscan.HDBSCAN(metric="l2", cluster_selection_epsilon=cluster_epsilon, prediction_data=True,
                                         min_samples=1)
        self.pca = PCA(n_components=n_components)

    def get_centroid(self, vectors, predicted_labels, cluster_num):

        cluster_info = []
        centroid = {}

        for cluster_id in range(0, cluster_num - 1, 1):
            idxs = np.where(predicted_labels == cluster_id)[0]
            cluster_vectors = vectors[np.expand_dims(idxs, axis=0)].squeeze(0)
            cluster_info.append(cluster_vectors)
            centroid[cluster_id] = np.mean(cluster_vectors, axis=0)

        return centroid

    def fit(self, X):

        # pca
        principalComponents = self.pca.fit_transform(X)
        noise = np.array(
            [np.random.normal(0, .01, (principalComponents.shape[1],)) for _ in range(principalComponents.shape[0])])
        principalComponents = principalComponents + noise

        self.clusterer.fit(principalComponents)
        train_labels = self.clusterer.labels_
        self.cltrid2centroid = self.get_centroid(principalComponents, train_labels, len(set(train_labels.tolist())))

        # return train_labels

    def transform(self, X):

        test_principalcomponents = self.pca.transform(X)
        test_labels, _ = hdbscan.approximate_predict(self.clusterer, test_principalcomponents)

        # output: the np.array(cluster_embeddings)
        cluster_represntation = np.array([self.cltrid2centroid[cluster_id] if (
                    cluster_id != -1 and cluster_id < len(self.cltrid2centroid) - 1) else test_principalcomponents[idx]
                                          for idx, cluster_id in enumerate(test_labels)])

        return cluster_represntation

    def sampler(self, templates, contents, sample_rate=0.2):

        sampled_templates = []

        sample_count_dict = Counter(templates)
        sample_count_dict = {k: int(v * sample_rate) + 1 for k, v in sample_count_dict.items()}

        for template, content in zip(templates, contents):
            if sample_count_dict[template] > 0:
                sampled_templates.append(content)
                sample_count_dict[template] -= 1

        return sampled_templates


class EVLOG_Dataset():

    def __init__(self, data_path, train_folder, evolved_folder, window_size=8):

        self.meta_data = {}
        self.meta_data['window_size'] = window_size
        self.pretrain_path = None
        self.meta_data["pretrain_matrix"] = None
        self.semcluster = SemanticCluster()

        self.train_data, self.test_data, self.transfer_data = self.load_pickle(
            root=data_path, train_folder=train_folder,
            trans_folder=evolved_folder)

        train_data = self.fit_and_transform(self.train_data, mode='train')
        test_data = self.fit_and_transform(self.test_data, mode='test')
        trans_data = self.fit_and_transform(self.transfer_data, mode='test')

        self.train_set = MyLOGAD(train_data)
        self.test_set = MyLOGAD(test_data)
        self.trans_set = MyLOGAD(trans_data)


    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 1) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        trans_loader = DataLoader(dataset=self.trans_set, batch_size=batch_size, shuffle=shuffle_test,
                                  num_workers=num_workers)

        return train_loader, test_loader, trans_loader


    def fit_and_transform(self, data, mode=None):
        logging.info("Fitting and Transforming data.")

        total_logs = []
        total_templates = []
        for each_data in data:
            total_logs.append(each_data["content"])
            total_templates.append(each_data["template"])

        if mode == 'train':
            self.sem_model = SimCSE("src/LPLM/my-sup-simcse-bert-base-uncased")
            self.ulog_train = set(total_logs)
            self.utemplate_train = set(total_templates)

        if mode == 'test':
            print('{} Unseen templates'.format(set(total_templates) - set(self.utemplate_train)))


        # unitary
        ucontents = list(set(total_logs))
        embs = self.sem_model.encode(ucontents, return_numpy=True, batch_size=256)
        clemb2idx = {log: embs[idx] for idx, log in enumerate(ucontents)}
        clemb2idx["PADDING"] = np.zeros(embs.shape[1]).reshape(-1)
        print('We have {} different unitary feature'.format(len(clemb2idx) - 1))
        logging.info("Extracting unitary features")

        # local
        if mode == 'train':
            print('HDBScan start training...')
            sampled_logs = self.semcluster.sampler(total_templates, total_logs)
            local_embs = self.sem_model.encode(sampled_logs, return_numpy=True, batch_size=256)
            self.semcluster.fit(local_embs)
        print('Extracting local features...')
        local_embs = embs
        cluster_embs = self.semcluster.transform(local_embs)
        clusteremb2idx = {log: cluster_embs[idx] for idx, log in enumerate(ucontents)}
        clusteremb2idx["PADDING"] = np.zeros(cluster_embs.shape[1]).reshape(-1)

        # global
        print('Extracting global features...')
        sort_freq = [i[0] for i in sorted(Counter(total_templates).items(), key=lambda x: x[1])]
        self.PARTITION = 10
        self.SESSION_LEN_DIM = 5
        batch_size = (len(sort_freq) / self.PARTITION) + 1
        template_freq = {template: int(idx / batch_size) for idx, template in enumerate(sort_freq)}

        prep_data = []
        for each_data in tqdm(data):
            content = clemb2idx[each_data["content"]]  # unitary
            localfeatures = self.extract_local_features(each_data["local_feature"], clusteremb2idx)
            globalfeatures = self.extract_global_features(each_data["session_templates"], each_data["session_len"],
                                                          template_freq)
            temp_prep_data = {}
            temp_prep_data['unitary_feature'] = np.array(content)
            temp_prep_data['content'] = each_data["content"]
            temp_prep_data['local_feature'] = localfeatures
            temp_prep_data['global_feature'] = globalfeatures
            temp_prep_data['label'] = each_data["label"]
            temp_prep_data["template"] = each_data['template']
            prep_data.append(temp_prep_data)

        self.meta_data['global_dim'] = self.PARTITION + self.SESSION_LEN_DIM
        self.meta_data['unitary_dim'] = 384
        self.meta_data['local_dim'] = 50

        return prep_data


    def generate_windows(self, data_dict):

        # input session, output windows
        # self.window_size
        windows = []
        pre_window_size = int(self.meta_data["window_size"] / 2)
        post_window_size = self.meta_data["window_size"] - pre_window_size
        data_dict["padding_templates"] = ["PADDING"] * pre_window_size + data_dict["Content"] + [
            "PADDING"] * post_window_size
        for idx in range(pre_window_size, pre_window_size + len(data_dict["Content"]), 1):
            tmp_window = data_dict["padding_templates"][idx - pre_window_size:idx + post_window_size]
            windows.append(tmp_window)

        return windows

    def extract_local_features(self, windows, log2idx):

        features = [log2idx[i] for i in windows]
        return np.array(features)

    def extract_global_features(self, templates, session_len, template_freq):
        # with two features: session length and templates frequency.

        # session length
        session_len_feats = [0] * self.SESSION_LEN_DIM
        if session_len < 10:
            session_len_feats[:1] = [1]
        elif session_len < 100:
            session_len_feats[:2] = [1] * 2
        elif session_len < 1000:
            session_len_feats[:3] = [1] * 3
        elif session_len < 10000:
            session_len_feats[:4] = [1] * 4
        else:
            session_len_feats[:5] = [1] * 5
        # session_len_feats = np.array(session_len_feats)
        session_len_feats = np.array(session_len_feats) / np.sqrt(np.sum(np.square(np.array(session_len_feats))))

        # templates frequency:
        template_freq_feats = [0] * self.PARTITION
        for template in templates:
            template_freq_feats[template_freq[template]] += 1
        template_freq_feats = np.array(template_freq_feats) / np.sqrt(np.sum(np.square(np.array(template_freq_feats))))

        return np.concatenate((session_len_feats, template_freq_feats), -1)

    def load_pickle(self, root, train_folder, trans_folder=None):

        with open(os.path.join(root, train_folder, 'session_train.pkl'), 'rb') as f:
            train_data = pickle.load(f)

        with open(os.path.join(root, train_folder, 'session_test.pkl'), 'rb') as f:
            test_data = pickle.load(f)

        with open(os.path.join(root, trans_folder, 'session_test.pkl'), 'rb') as f:
            trans_data = pickle.load(f)


        train_sessions = []
        for session_id, data_dict in train_data.items():
            windows = self.generate_windows(data_dict)
            if data_dict["label"] == 0:  # normal window
                for t, c, window, label in zip(data_dict["templates"], data_dict["Content"], windows,
                                               data_dict["rca_label"]):
                    tmp_train_session = {}
                    tmp_train_session["template"] = t
                    tmp_train_session["content"] = c
                    tmp_train_session["label"] = label
                    tmp_train_session["local_feature"] = window
                    tmp_train_session["session_len"] = len(data_dict["templates"])
                    tmp_train_session["session_templates"] = data_dict["templates"]
                    train_sessions.append(tmp_train_session)


        test_sessions = []
        for session_id, data_dict in test_data.items():
            # if data_dict["label"] == 0: # normal window
            windows = self.generate_windows(data_dict)
            for t, c, window, label in zip(data_dict["templates"], data_dict["Content"], windows,
                                           data_dict["rca_label"]):
                # label = template_gt[t]
                tmp_test_session = {}
                tmp_test_session["template"] = t
                tmp_test_session["content"] = c
                tmp_test_session["label"] = label
                tmp_test_session["local_feature"] = window
                tmp_test_session["session_len"] = len(data_dict["templates"])
                tmp_test_session["session_templates"] = data_dict["templates"]
                test_sessions.append(tmp_test_session)
                # test_sessions.append((t, label))

        trans_sessions = []
        for session_id, data_dict in trans_data.items():
            # if data_dict["label"] == 0: # normal window
            windows = self.generate_windows(data_dict)
            for t, c, window, label in zip(data_dict["templates"], data_dict["Content"], windows,
                                           data_dict["rca_label"]):
                # label = template_gt[t]
                tmp_trans_session = {}
                tmp_trans_session["template"] = t
                tmp_trans_session["content"] = c
                tmp_trans_session["label"] = label
                tmp_trans_session["local_feature"] = window
                tmp_trans_session["session_len"] = len(data_dict["templates"])
                tmp_trans_session["session_templates"] = data_dict["templates"]
                trans_sessions.append(tmp_trans_session)
                # test_sessions.append((t, label))

        return train_sessions, test_sessions, trans_sessions


class MyLOGAD(Dataset):

    def __init__(self, prep_data):

        flatten_data_list = []
        for idx, each_data in enumerate(prep_data):
            flatten_sample = (torch.from_numpy(each_data["unitary_feature"]).unsqueeze(0), each_data["label"], idx,
                              torch.from_numpy(each_data["local_feature"]).unsqueeze(0),
                              torch.from_numpy(each_data["global_feature"]).unsqueeze(0))
            flatten_data_list.append(flatten_sample)
        self.flatten_data_list = flatten_data_list

        visualized_list = []
        for idx, each_data in enumerate(prep_data):
            flatten_sample = (each_data["template"], each_data["label"], idx)
            visualized_list.append(flatten_sample)
        self.visualized_list = visualized_list

    def retrieve(self, id):
        return self.visualized_list[id]

    def __len__(self):
        return len(self.flatten_data_list)

    def __getitem__(self, index):
        sample = self.flatten_data_list[index]
        return sample[0], sample[1], sample[2], sample[3], sample[4]