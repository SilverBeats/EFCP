import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import SequenceSummary

class AttentionModule(nn.Module):
    def __init__(self, k: int, d: int):
        super(AttentionModule, self).__init__()
        self.wx = nn.Linear(d, k)
        self.wg = nn.Linear(d, k)
        self.whx = nn.Linear(k, 1)

    def forward(self, x, g):
        # x.shape = (bs, t, d)
        # g.shape = (bs, 1, d)
        assert x.shape[0] == g.shape[0]
        bs, t, d = x.shape
        ones = torch.ones(size=(bs, t, 1), device=x.device).float()
        H = torch.tanh(self.wx(x) + torch.bmm(ones, self.wg(g)))  # H.shape = (bs, t, k)
        w_atten = F.softmax(self.whx(H), dim=-1)  # w_atten.shape = (bs, t, 1)
        return (w_atten * x).sum(dim=1)  # (bs, d)


class CoAttention(nn.Module):
    def __init__(self, k: int, d: int):
        super(CoAttention, self).__init__()
        self.k = k
        self.d = d
        self.step_one_attention = AttentionModule(k, d)
        self.step_two_attention = AttentionModule(k, d)
        self.step_third_attention = AttentionModule(k, d)

    def forward(self, x, y):
        bs = x.shape[0]
        g = torch.zeros(size=(bs, 1, self.d), device=x.device)
        # first attention step
        s_hat = self.step_one_attention(x, g)
        # second attention step
        y_hat = self.step_two_attention(y, s_hat.unsqueeze(1))
        # third attention step
        x_hat = self.step_third_attention(x, y_hat.unsqueeze(1))
        return x_hat, y_hat


class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(MLP, self).__init__()

        self.lin_1 = nn.Linear(input_dim, hid_dim, bias=False)
        self.lin_2 = nn.Linear(hid_dim, out_dim, bias=False)

        self.act = nn.ELU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)

        return x


class FuseLinearModule(nn.Module):
    def __init__(self, config, fuse_type, use_persona=True):
        super(FuseLinearModule, self).__init__()
        self.ctx_choice_head = SequenceSummary(config)
        self.use_persona = use_persona
        if self.use_persona:
            self.persona_choice_head = SequenceSummary(config)
            hs = config.n_embd
            self.cp_linear = nn.Sequential(nn.Linear(2 * hs, hs), nn.ELU())

    def forward(self, ctx_h_s, ctx_len, **kwargs):
        ctx_last_token_hidden_states = self.ctx_choice_head(ctx_h_s, ctx_len - 1)

        if self.use_persona:
            persona_len = kwargs.get('persona_len')
            p_h_s = kwargs.get('p_h_s')
            persona_last_token_hidden_states = self.persona_choice_head(p_h_s, kwargs.get('persona_len') - 1)
            t = self.cp_linear(torch.cat([ctx_last_token_hidden_states, persona_last_token_hidden_states], dim=-1))
        else:
            t = ctx_last_token_hidden_states

        return t


class FuseCoAttentionModule(nn.Module):
    def __init__(self, k, hs):
        super(FuseCoAttentionModule, self).__init__()
        self.co = CoAttention(k, hs)
        self.tw = torch.nn.Parameter(torch.ones(2, 1) / 2)

    def forward(self, ctx_h_s, p_h_s):
        # (bs, hs); (bs, hs)
        c, p = self.co(ctx_h_s, p_h_s)
        # (2, bs, hs)
        t = torch.stack([c, p], dim=0) * self.tw.unsqueeze(-1)
        r = torch.mean(t, dim=0)
        # (bs, hs)
        return r


class CommonSenseModule(nn.Module):
    def __init__(self, gpt2_config, encoder_config, **kwargs):
        super(CommonSenseModule, self).__init__()
        self.hs = hs = gpt2_config.n_embd

        self.multi_choose_head = SequenceSummary(gpt2_config)

        self.tw = nn.Parameter(torch.ones(6, 1) / 6)

    def forward(self, ctx_h_s, ctx_token_type_ids, ctx_cs_dict, pad, gpt2):
        token_type_id = ctx_token_type_ids[:, 0].unsqueeze(1)  # (bs, 1)
        batch_size = ctx_h_s.shape[0]
        device = ctx_h_s.device
        idxs = torch.arange(0, batch_size, device=device).long()

        cs = []
        loss_arr = []
        # encode the common sense sentences
        for key, value in ctx_cs_dict.items():
            attention_mask = value.ne(pad).float()  # (bs, len)
            position_ids = torch.cumsum(attention_mask, dim=-1).long() - 1
            token_type_ids = token_type_id.expand(-1, value.shape[1])  # (bs, len)
            true_len = attention_mask.sum(-1).long()  # bs. exclude pad
            # (bs, len, hs)
            outputs = gpt2.encode(input_ids=value,
                                  labels=value,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids)
            value_h_s = outputs[0]
            loss_arr.append(outputs[1])
            if 'xReact' in key or 'xAttr' in key:
                # (bs, hs)
                value_h_s = torch.stack([torch.mean(value_h_s[i, :l, :], dim=0) for i, l in zip(idxs, true_len)])
                # (bs, ctx_len, hs)
                value_h_s = value_h_s.unsqueeze(1).expand(-1, ctx_h_s.shape[1], -1)
            else:
                # (bs, ctx_len, hs)
                value_h_s = self.multi_choose_head(value_h_s, true_len - 1).unsqueeze(1).expand(-1, ctx_h_s.shape[1],
                                                                                                -1)
            cs.append(value_h_s)

        loss = sum(loss_arr)
        # (6, bs, ctx_len, hs)
        scale_tensor = torch.stack([ctx_h_s] + cs, dim=0)
        ctx_h_s = torch.sum(scale_tensor * self.tw.view(6, 1, 1, 1), dim=0)

        return loss, ctx_h_s


class PredictDialogueFactorsModule(nn.Module):
    def __init__(self, hs):
        super(PredictDialogueFactorsModule, self).__init__()
        self.er_embeddings = nn.Embedding(2, hs)
        self.ex_embeddings = nn.Embedding(2, hs)
        self.in_embeddings = nn.Embedding(2, hs)
        self.dialact_embeddings = nn.Embedding(9, hs)
        self.emotion_embeddings = nn.Embedding(10, hs)
        # std is equal to gpt2.config.initializer_range
        self.er_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.ex_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.in_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.dialact_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.emotion_embeddings.weight.data.normal_(mean=0.0, std=0.02)

        self.er_head = nn.Sequential(nn.Linear(hs, hs), nn.ELU())
        self.ex_head = nn.Sequential(nn.Linear(hs, hs), nn.ELU())
        self.in_head = nn.Sequential(nn.Linear(hs, hs), nn.ELU())
        self.da_head = nn.Sequential(nn.Linear(2 * hs, hs), nn.ELU())
        self.em_head = nn.Sequential(nn.Linear(3 * hs, hs), nn.ELU())

    def get_ctx_df(self, ctx_da, ctx_em):
        ctx_additive_embeds = self.dialact_embeddings(ctx_da) + \
                              self.emotion_embeddings(ctx_em)
        return ctx_additive_embeds

    def forward(self, t, tgt_er, tgt_ex, tgt_in, tgt_da, tgt_em, result_info):
        self.pred_cm(t, tgt_er, tgt_ex, tgt_in, result_info)
        self.pred_da(t, tgt_da, result_info)
        self.pred_em(t, tgt_em, result_info)

    def pred_cm(self, l_h_s, tgt_er, tgt_ex, tgt_in, result_info):
        tgt_er_logits = F.linear(self.er_head(l_h_s), self.er_embeddings.weight)
        tgt_ex_logits = F.linear(self.ex_head(l_h_s), self.ex_embeddings.weight)
        tgt_in_logits = F.linear(self.in_head(l_h_s), self.in_embeddings.weight)
        # calc loss
        pred_er_loss = F.cross_entropy(tgt_er_logits, tgt_er.view(-1), reduction='mean')  # scalar
        pred_ex_loss = F.cross_entropy(tgt_ex_logits, tgt_ex.view(-1), reduction='mean')  # scalar
        pred_in_loss = F.cross_entropy(tgt_in_logits, tgt_in.view(-1), reduction='mean')  # scalar

        # get the pred result
        pred_er_top1 = torch.topk(tgt_er_logits, k=1, dim=-1)[1]
        pred_ex_top1 = torch.topk(tgt_ex_logits, k=1, dim=-1)[1]
        pred_in_top1 = torch.topk(tgt_in_logits, k=1, dim=-1)[1]

        er_embeds = self.er_embeddings(tgt_er.view(-1))  # (bs, hs)
        ex_embeds = self.ex_embeddings(tgt_ex.view(-1))  # (bs, hs)
        in_embeds = self.in_embeddings(tgt_in.view(-1))  # (bs, hs)

        cm_embeds = er_embeds + ex_embeds + in_embeds
        cm_loss = pred_er_loss + pred_ex_loss + pred_in_loss
        result_info.update({
            'pred_er': tgt_er,
            'pred_ex': tgt_ex,
            'pred_in': tgt_in,
            'pred_er_top1': pred_er_top1,
            'pred_in_top1': pred_in_top1,
            'pred_ex_top1': pred_ex_top1,
            'cm_embeds': cm_embeds,
            'cm_loss': cm_loss
        })

    def pred_da(self, l_h_s, tgt_da, result_info):
        tgt_da_logits = F.linear(self.da_head(torch.cat([
            l_h_s,
            result_info.get('cm_embeds')
        ], dim=-1)), self.dialact_embeddings.weight)
        pred_da_loss = F.cross_entropy(tgt_da_logits, tgt_da.view(-1), reduction='mean')  # scalar
        pred_da_top1 = torch.topk(tgt_da_logits, k=1, dim=-1)[1]
        pred_da_top3 = torch.topk(tgt_da_logits, k=3, dim=-1)[1]

        da_embeds = self.dialact_embeddings(tgt_da.view(-1))  # (bs, hs)

        result_info.update({
            'pred_da': tgt_da,
            'pred_da_top1': pred_da_top1,
            'pred_da_top3': pred_da_top3,
            'da_loss': pred_da_loss,
            'da_embeds': da_embeds
        })

    def pred_em(self, l_h_s, tgt_em, result_info):
        tgt_em_logits = F.linear(self.em_head(torch.cat([
            l_h_s,
            result_info.get('cm_embeds'),
            result_info.get('da_embeds')
        ], dim=-1)), self.emotion_embeddings.weight)
        pred_em_loss = F.cross_entropy(tgt_em_logits, tgt_em.view(-1), reduction='mean')  # scalar
        pred_em_top1 = torch.topk(tgt_em_logits, k=1, dim=-1)[1]
        pred_em_top3 = torch.topk(tgt_em_logits, k=3, dim=-1)[1]

        em_embeds = self.emotion_embeddings(tgt_em.view(-1))  # (bs, hs)

        result_info.update({
            'pred_em': tgt_em,
            'pred_em_top1': pred_em_top1,
            'pred_em_top3': pred_em_top3,
            'em_loss': pred_em_loss,
            'em_embeds': em_embeds
        })


