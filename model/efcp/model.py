import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module import CommonSenseModule, FuseCoAttentionModule, FuseLinearModule, PredictDialogueFactorsModule
from modeling.gpt2_2 import GPT2EncoderDecoderModel
from utils.generate import generate


class EFCP(nn.Module):
    def __init__(self, gpt2: GPT2EncoderDecoderModel, token_id_dict, **kwargs):
        super(EFCP, self).__init__()
        self.pad = token_id_dict['pad']
        self.gpt2 = gpt2
        self.config = gpt2.config
        self.hs = hs = gpt2.config.n_embd

        self.pred_comae_coef = kwargs.get('pred_comae_coef', 0)
        self.encode_lm_coef = kwargs.get('encode_lm_coef', 0)

        # ablation_config
        ablation_config = kwargs.get('ablation_config')
        self.do_acs = ablation_config.get('do_acs', True)
        self.use_persona = ablation_config.get('use_persona', True)
        self.pred_with_persona = ablation_config.get('pred_with_persona', True)
        self.pdf = PredictDialogueFactorsModule(hs)

        if self.do_acs:
            encoder_config = kwargs.get('encoder_config')
            self.acs = CommonSenseModule(gpt2_config=gpt2.config,
                                         encoder_config=encoder_config)
        self.fuse_c_p_type = kwargs.get('fuse_c_p_type')
        self.fuse_module = None  # will create by set_fuse_module

    def forward(self,
                ctx_input_ids, ctx_token_type_ids, ctx_da, ctx_em, ctx_len,
                ctx_xReact, ctx_xEffect, ctx_xWant, ctx_xIntent, ctx_xNeed,
                tgt_input_ids, tgt_label_ids, tgt_token_type_ids,
                tgt_da, tgt_em, tgt_er, tgt_ex, tgt_in, tgt_len,
                persona_input_ids, persona_token_type_ids, persona_len,
                do_eval=False, do_gen=False, **kwargs):
        """
        Args:
            ctx_input_ids (torch.Tensor):  (bs, ctx_len)
            ctx_token_type_ids (torch.Tensor):  (bs, ctx_len)
            ctx_da (torch.Tensor): (bs, 1)
            ctx_em (torch.Tensor): (bs, 1)
            ctx_len (torch.Tensor): bs
            ctx_xReact (torch.Tensor): (bs, len)
            ctx_xEffect (torch.Tensor): (bs, len)
            ctx_xWant (torch.Tensor): (bs, len)
            ctx_xIntent (torch.Tensor): (bs, len)
            ctx_xNeed (torch.Tensor): (bs, len)
            tgt_input_ids (torch.Tensor): (bs, tgt_len)
            tgt_label_ids (torch.Tensor): (bs, tgt_len)
            tgt_token_type_ids (torch.Tensor): (bs, tgt_len)
            tgt_da (torch.Tensor): (bs, 1)
            tgt_em (torch.Tensor): (bs, 1)
            tgt_er (torch.Tensor): (bs, 1)
            tgt_ex (torch.Tensor): (bs, 1)
            tgt_in (torch.Tensor): (bs, 1)
            tgt_len (torch.Tensor): bs
            persona_input_ids: (bs, num_p, p_len)
            persona_token_type_ids (torch.Tensor):  (bs, num_p, p_len)
            persona_len (torch.Tensor): (bs, num_p)
            do_eval (bool):
            do_gen (bool):
            **kwargs (dict):
        """
        result_info = dict()
        bs = ctx_input_ids.shape[0]
        idxs = torch.arange(0, bs).type_as(ctx_input_ids)

        # (bs, 2 + ctx_len, hs)
        ctx_h_s, encode_ctx_loss = self.encode_ctx(ctx_input_ids, ctx_token_type_ids, ctx_da, ctx_em)

        if self.do_acs:
            cs_dict = {
                'ctx_xIntent': ctx_xIntent,
                'ctx_xWant': ctx_xWant,
                'ctx_xNeed': ctx_xNeed,
                'ctx_xReact': ctx_xReact,
                'ctx_xEffect': ctx_xEffect
            }
            ecs_loss, ctx_h_s = self.acs(ctx_h_s, ctx_token_type_ids, cs_dict, self.pad, self.gpt2)
        if self.use_persona:
            # (bs, p_len, hs)
            p_h_s, encode_persona_loss = self.encode_persona(persona_input_ids, persona_token_type_ids)
            result_info.update({'ep': encode_persona_loss})
            if self.pred_with_persona:
                # (bs, hs)
                t = self.fuse(ctx_h_s=ctx_h_s, p_h_s=p_h_s, persona_len=persona_len, ctx_len=ctx_len)
            else:
                t = self.fuse(ctx_h_s=ctx_h_s, ctx_len=ctx_len)
        else:
            t = self.fuse(ctx_h_s=ctx_h_s, ctx_len=ctx_len)

        # predict the dialogue factors of response
        self.pdf(t, tgt_er, tgt_ex, tgt_in, tgt_da, tgt_em, result_info)

        enc_contexts = [ctx_h_s, p_h_s] if self.use_persona else [ctx_h_s]

        tgt_attention_mask = tgt_input_ids.ne(self.pad).float()
        tgt_position_ids = torch.cumsum(tgt_attention_mask, dim=-1).long() - 1
        # (bs, hs)
        tgt_additive_embeds = result_info.pop('cm_embeds') + \
                              result_info.pop('da_embeds') + \
                              result_info.pop('em_embeds')
        # (bs, tgt_len, bs)
        tgt_additive_embeds = tgt_additive_embeds.unsqueeze(1).expand(-1, tgt_input_ids.shape[1], -1)

        if do_gen:
            d = {
                'input_ids': tgt_input_ids,
                'additive_embeds': tgt_additive_embeds,
                'attention_mask': tgt_attention_mask,
                'position_ids': tgt_position_ids,
                'enc_contexts': enc_contexts,
                'token_type_ids': tgt_token_type_ids
            }
            return d, result_info

        lm_logits = self.gpt2.decode(input_ids=tgt_input_ids,
                                     additive_embeds=tgt_additive_embeds,
                                     token_type_ids=tgt_token_type_ids,
                                     enc_contexts=enc_contexts,
                                     position_ids=tgt_position_ids,
                                     attention_mask=tgt_attention_mask)[0]

        lm_loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), tgt_label_ids.view(-1),
                                  ignore_index=-1, reduction='none').view(tgt_label_ids.size(0), tgt_label_ids.size(1))
        label_size = torch.sum(tgt_label_ids.ne(-1), dim=1).type_as(lm_loss)
        lm_loss_value = torch.sum(lm_loss) / torch.sum(label_size)
        ppl_value = torch.exp(torch.mean(torch.sum(lm_loss, dim=1).float() / label_size.float()))

        if do_eval:
            return lm_loss, label_size

        return_dict = {
            'tcm': result_info.pop('cm_loss'),
            'tda': result_info.pop('da_loss'),
            'tem': result_info.pop('em_loss'),
        }

        enc_loss = encode_ctx_loss
        if self.do_acs:
            enc_loss = enc_loss + ecs_loss
            return_dict['ecs'] = ecs_loss
        if self.use_persona:
            ep = result_info.pop('ep')
            return_dict['ep'] = ep
            enc_loss = enc_loss + ep

        total_loss = lm_loss_value + self.pred_comae_coef * sum(return_dict.values()) + self.encode_lm_coef * enc_loss
        return_dict.update({'all': total_loss, 'ppl': ppl_value})
        return return_dict

    def set_fuse_module(self, **kwargs):  # used by build.py
        if self.fuse_c_p_type == 'linear':
            self.fuse_module = FuseLinearModule(self.config, self.fuse_c_p_type,
                                                self.use_persona and self.pred_with_persona)
        elif self.fuse_c_p_type == 'co':
            k = kwargs.get('k')
            self.fuse_module = FuseCoAttentionModule(k, self.hs)
        else:
            raise NotImplementedError()

    def fuse(self, **kwargs):
        if self.fuse_c_p_type == 'linear':
            ctx_h_s = kwargs.pop('ctx_h_s')
            ctx_len = kwargs.pop('ctx_len')
            t = self.fuse_module(ctx_h_s, ctx_len, **kwargs)
        elif self.fuse_c_p_type == 'co':
            ctx_h_s = kwargs.pop('ctx_h_s')
            p_h_s = kwargs.pop('p_h_s')
            t = self.fuse_module(ctx_h_s, p_h_s)
        else:
            raise NotImplementedError()
        return t

    def encode_ctx(self, ctx_input_ids, ctx_token_type_ids, ctx_da, ctx_em):
        ctx_additive_embeds = self.pdf.get_ctx_df(ctx_da, ctx_em)
        ctx_attention_mask = ctx_input_ids.ne(self.pad).float()
        ctx_position_ids = torch.cumsum(ctx_attention_mask, dim=-1).long() - 1
        ctx_h_s, loss = self.gpt2.encode(input_ids=ctx_input_ids,
                                         labels=ctx_input_ids,
                                         additive_embeds=ctx_additive_embeds,
                                         token_type_ids=ctx_token_type_ids,
                                         position_ids=ctx_position_ids,
                                         attention_mask=ctx_attention_mask)
        return ctx_h_s, loss

    def encode_persona(self, persona_input_ids, persona_token_type_ids):
        persona_attention_mask = persona_input_ids.ne(self.pad).float()
        persona_position_ids = torch.cumsum(persona_attention_mask, dim=-1).long() - 1
        p_h_s, loss = self.gpt2.encode(input_ids=persona_input_ids,
                                       labels=persona_input_ids,
                                       token_type_ids=persona_token_type_ids,
                                       attention_mask=persona_attention_mask,
                                       position_ids=persona_position_ids)
        return p_h_s, loss

    @torch.no_grad()
    def generate_step(self, *args, **kwargs):
        lm_logits, past_key_values, *_ = self.gpt2.decode(*args, **kwargs)
        return {
            'lm_logits': lm_logits,
            'past_key_values': past_key_values,
        }

    @torch.no_grad()
    def generate(self, forward_dict, **kwargs):
        d, result_info = self.forward(**forward_dict, do_gen=True)
        return result_info, generate(self, **d, **kwargs)
