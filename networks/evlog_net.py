import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from base.base_net import BaseNet


class AttentionalFlow(nn.Module):
    def __init__(self, input_size):
        super(AttentionalFlow, self).__init__()
        self.d = input_size//2
        self.W = nn.Linear(6 * self.d, 1, bias=False)


    def forward(self, contx, query, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"

        query=query
        key=contx
        value=contx

        d_k = query.size(-1) #e
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k) # [b*1*e], [b*e*8]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        weighted = torch.matmul(p_attn, value).squeeze()
        attn_weight = p_attn

        # print(p_attn.size())

        return weighted, attn_weight



# merge net
class EVLOG_Net(BaseNet):

    def __init__(self, meta_data):
        super().__init__()
        unitary_dim = meta_data['unitary_dim']
        local_dim = meta_data['local_dim']
        template_embedding_dim = 32


        self.feat_dim = 32
        # self.rep_dim = 2 * self.feat_dim + 8
        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        # local
        # self.conv_trans = nn.Conv1d(local_dim, 32, 3, bias=False, padding=1)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(96, 32, bias=False)

        #template
        self.template_fc1 = nn.Linear(unitary_dim, 128, bias=False)
        self.template_fc2 = nn.Linear(128, self.feat_dim, bias=False)
        self.template_bn1 = nn.BatchNorm1d(unitary_dim, eps=1e-04, affine=False)
        self.template_bn2 = nn.BatchNorm1d(128, eps=1e-04, affine=False)


        #global feature
        global_feat_size = meta_data['global_dim']
        self.global_bn = nn.BatchNorm1d(global_feat_size, eps=1e-04, affine=False)
        self.global_fc1 = nn.Linear(global_feat_size, 16, bias=False)
        self.global_fc2 = nn.Linear(16, 32, bias=False)


        #attention-flow
        self.attn_flow_fc1 = nn.Linear(50, 128, bias=False)
        self.attn_flow_bn1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.attn_flow_fc2 = nn.Linear(128, self.feat_dim, bias=False)
        self.attn_flow = AttentionalFlow(50)
        self.attn_flow_cl_fc1 = nn.Linear(unitary_dim, 128, bias=False)
        self.attn_flow_cl_bn1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)



        # local in weighted loss
        # self.weighted_loss = torch.tensor([0.1, 0.3, 1, 0.3, 0.1])

    def forward(self, x, local_feat, global_feat):
        local_feat = local_feat.float()
        global_feat = global_feat.float()
        x = x.float()


        unitary_embedding = self.attn_flow_cl_fc1(x.squeeze())
        unitary_embedding = F.leaky_relu(self.attn_flow_cl_bn1(unitary_embedding)).unsqueeze(1)
        batch_size = local_feat.size(0)
        local_feat = local_feat.squeeze(1)
        window_size = local_feat.size(1)
        template_size = local_feat.size(2)
        tem_embedding = local_feat.view(-1, template_size)
        tem_embedding = self.attn_flow_fc1(tem_embedding)
        tem_embedding = F.leaky_relu(self.attn_flow_bn1(tem_embedding))
        tem_embedding = tem_embedding.view(batch_size, window_size, 128)
        weighted_embedding, _ = self.attn_flow(tem_embedding, unitary_embedding)
        # print(weighted_embedding.size())
        # weighted_embedding = F.leaky_relu(self.attn_flow_bn1(self.attn_flow_fc1(weighted_embedding)))
        weighted_embedding = self.attn_flow_fc2(weighted_embedding)



        # single
        sing_embedding = x
        sing_embedding = sing_embedding.view(sing_embedding.size(0), -1)
        sing_embedding = F.leaky_relu(self.template_bn1(sing_embedding))
        sing_embedding = self.template_fc2(F.leaky_relu(self.template_bn2(self.template_fc1(sing_embedding))))

        # # global, which we do not use actually.
        global_embedding = global_feat.view(global_feat.size(0),-1)
        global_embedding = F.leaky_relu(self.global_bn(global_embedding))
        global_embedding = self.global_fc2(self.global_fc1(global_embedding))


        return sing_embedding, weighted_embedding, global_embedding
