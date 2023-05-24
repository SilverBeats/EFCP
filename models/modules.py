from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ACT2CLS = {
    'elu': nn.ELU,
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
}


def gather_by_len(input_tensor: Tensor, seq_len: Tensor):
    seq_len = seq_len.view(-1)
    assert seq_len.shape[0] == input_tensor.shape[0]
    arr = []  # [(hs, ),...]
    for t, l in zip(input_tensor, seq_len):
        arr.append(t[l - 1])
    return torch.stack(arr, dim=0)  # (bs, hs)


def _ef_embedding_wrapper(layer: nn.Module, input_ids: Tensor) -> Tensor:
    original_shape = input_ids.shape
    embeds = layer(input_ids.reshape(-1))
    return embeds.reshape(*original_shape, -1)


class PredictEmpathyFactorsModule(nn.Module):
    def __init__(self, hidden_size: int, act: str):
        super(PredictEmpathyFactorsModule, self).__init__()
        self.er_embeddings = nn.Embedding(2, hidden_size)
        self.ip_embeddings = nn.Embedding(2, hidden_size)
        self.ex_embeddings = nn.Embedding(2, hidden_size)
        self.da_embeddings = nn.Embedding(9, hidden_size)
        self.em_embeddings = nn.Embedding(10, hidden_size)

        self.er_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.ip_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.ex_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.da_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.em_embeddings.weight.data.normal_(mean=0.0, std=0.02)

        act_fn = ACT2CLS[act]

        self.er_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), act_fn())
        self.ip_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), act_fn())
        self.ex_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), act_fn())
        self.da_head = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), act_fn())
        self.em_head = nn.Sequential(nn.Linear(3 * hidden_size, hidden_size), act_fn())

    def embed_cm(
            self,
            er: Tensor = None,
            ip: Tensor = None,
            ex: Tensor = None
    ):
        return_dict = {}
        if er is not None:
            return_dict['er'] = _ef_embedding_wrapper(self.er_embeddings, er)
        if ip is not None:
            return_dict['ip'] = _ef_embedding_wrapper(self.ip_embeddings, ip)
        if ex is not None:
            return_dict['ex'] = _ef_embedding_wrapper(self.ex_embeddings, ex)
        return return_dict

    def embed_da(self, da: Tensor):
        return _ef_embedding_wrapper(self.da_embeddings, da)

    def embed_em(self, em: Tensor):
        return _ef_embedding_wrapper(self.em_embeddings, em)

    def pred_cm(
            self,
            hidden_state: Tensor,
            er: Tensor,
            ip: Tensor,
            ex: Tensor,
            result_info: Dict[str, Any]
    ) -> Tensor:
        r"""
        predict response's cm(er, ip, ex) based on the hidden_state

        hidden_state.shape = (bs, hs)
        er.shape = (bs, ) or (bs, 1)
        ip.shape = (bs, ) or (bs, 1)
        ex.shape = (bs, ) or (bs, 1)
        """
        # predict
        er_logits = F.linear(self.er_head(hidden_state), self.er_embeddings.weight)
        ip_logits = F.linear(self.ip_head(hidden_state), self.ip_embeddings.weight)
        ex_logits = F.linear(self.ex_head(hidden_state), self.ex_embeddings.weight)

        pred_er_top1 = torch.topk(er_logits, k=1, dim=-1)[1]
        pred_ip_top1 = torch.topk(ip_logits, k=1, dim=-1)[1]
        pred_ex_top1 = torch.topk(ex_logits, k=1, dim=-1)[1]

        # calc loss
        pred_er_loss = F.cross_entropy(er_logits, er.view(-1), reduction='mean')  # scalar
        pred_ip_loss = F.cross_entropy(ip_logits, ip.view(-1), reduction='mean')  # scalar
        pred_ex_loss = F.cross_entropy(ex_logits, ex.view(-1), reduction='mean')  # scalar
        pred_cm_loss = pred_er_loss + pred_ip_loss + pred_ex_loss

        # embed the ground labels
        er_embeds = self.er_embeddings(er.view(-1))  # (bs, hidden_size)
        ip_embeds = self.ip_embeddings(ip.view(-1))  # (bs, hidden_size)
        ex_embeds = self.ex_embeddings(ex.view(-1))  # (bs, hidden_size)
        cm_embeds = er_embeds + ip_embeds + ex_embeds

        # save the result
        result_info.update({
            'pred_er': er,
            'pred_ip': ip,
            'pred_ex': ex,
            'pred_er_top1': pred_er_top1,
            'pred_ip_top1': pred_ip_top1,
            'pred_ex_top1': pred_ex_top1,
            'cm_embeds': cm_embeds,
        })
        return pred_cm_loss

    def pred_da(
            self,
            hidden_state: Tensor,
            da: Tensor,
            result_info: Dict[str, Any]
    ) -> Tensor:
        r"""
        predict response's dialog act label based on hidden_state and cm embeds
        [hidden_state; cm_embeds] -> da

        hidden_state.shape = (bs, hidden_size)
        da.shape = (bs, ) or (bs, 1)
        """

        da_logits = F.linear(self.da_head(torch.cat((
            hidden_state,
            result_info.get('cm_embeds')
        ), dim=-1)), self.da_embeddings.weight)

        pred_da_top1 = torch.topk(da_logits, k=1, dim=-1)[1]
        pred_da_top3 = torch.topk(da_logits, k=3, dim=-1)[1]

        # calc predict loss
        pred_da_loss = F.cross_entropy(da_logits, da.view(-1), reduction='mean')  # scalar

        # embed the ground label
        da_embeds = self.da_embeddings(da.view(-1))  # (bs, hidden_size)

        # save result
        result_info.update({
            'pred_da': da,
            'pred_da_top1': pred_da_top1,
            'pred_da_top3': pred_da_top3,
            'da_embeds': da_embeds
        })
        return pred_da_loss

    def pred_em(
            self,
            hidden_state: Tensor,
            em: Tensor,
            result_info: Dict[str, Any]
    ) -> Tensor:
        r"""
        predict response's emotion label based on hidden_state, cm embeds and da embeds
        [hidden_state; cm_embeds; da_embeds] -> em

        hidden_state.shape = (bs, hs)
        em.shape = (bs, 1) or (bs, )
        """

        em_logits = F.linear(self.em_head(torch.cat((
            hidden_state,
            result_info.get('cm_embeds'),
            result_info.get('da_embeds')
        ), dim=-1)), self.em_embeddings.weight)

        pred_em_top1 = torch.topk(em_logits, k=1, dim=-1)[1]
        pred_em_top3 = torch.topk(em_logits, k=3, dim=-1)[1]

        # calc predict loss
        pred_em_loss = F.cross_entropy(em_logits, em.view(-1), reduction='mean')  # scalar

        # embed the ground label
        em_embeds = self.em_embeddings(em.view(-1))  # (bs, hidden_size)

        # save result
        result_info.update({
            'pred_em': em,
            'pred_em_top1': pred_em_top1,
            'pred_em_top3': pred_em_top3,
            'em_embeds': em_embeds
        })
        return pred_em_loss

    def forward(
            self,
            hidden_state: Tensor,
            er: Tensor,
            ip: Tensor,
            ex: Tensor,
            da: Tensor,
            em: Tensor
    ) -> Dict[str, Any]:
        result_info = dict()
        pred_cm_loss = self.pred_cm(hidden_state, er, ip, ex, result_info)
        pred_da_loss = self.pred_da(hidden_state, da, result_info)
        pred_em_loss = self.pred_em(hidden_state, em, result_info)

        pred_loss = pred_cm_loss + pred_da_loss + pred_em_loss
        result_info['loss'] = pred_loss
        return result_info


class MLP(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int, out_dim: int, act: str):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hid_dim, bias=False),
            ACT2CLS[act](),
            nn.Linear(hid_dim, out_dim, bias=False)
        )

    def forward(self, x):
        return self.mlp(x)
