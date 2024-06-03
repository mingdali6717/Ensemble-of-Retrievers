import torch
from torch import nn


## Note that the following code is modified from
## https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/utils/model/reward_model.py
class RewardModel(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0, need_decoder_input=False):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        self.need_decoder_input = need_decoder_input
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim, 1, bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):
        loss = None

        if self.need_decoder_input:
            decoder_input_ids = input_ids
            decoder_input_ids = self.rwtranrsformer._shift_right(decoder_input_ids)
            # decoder_input_ids.to(input_ids.device) 没什么用
            transformer_outputs = self.rwtranrsformer(
                input_ids = input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                decoder_input_ids = decoder_input_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache)
        else:
            if head_mask is None:
                transformer_outputs = self.rwtranrsformer(
                    input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    # head_mask=head_mask, LLama不能接收这个
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache)
            else:
                transformer_outputs = self.rwtranrsformer(
                    input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache)

        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]
            # 这个东西是这样的，一般都是Encoder-Decoder或者Decoder架构，取decoder部分的输出。而decoder部分的注意力计算方式决定了，一般都是取序列最后一个有效token。比如序列长120，pad到500，那么就应该取121位上的pad对应的值。所以一般也就是第一个pad，但是opt模型和其他的一些模型有一个不同点就是开头有一个pad，所以这个时候要略过第一个，就把pad_on_the_begining设为1，这样取得就是第二个，否则是0就取得是第一个。但是如果序列超过了最大长度，那么就无所谓了，直接取最后一个。
            c_inds = (chosen_id == self.PAD_ID).nonzero()
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds # 所以当序列超过最大长度的时候，就是else的情况，这时候直接最后一个。
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
                                                    r_truncated_reward).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }

    def forward_value(self, input_ids=None, attention_mask=None, prompt_length=0, 
                past_key_values=None, head_mask=None, inputs_embeds=None, use_cache=False):

        if head_mask is None:
            transformer_outputs = self.rwtranrsformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                # head_mask=head_mask, LLama不能接收这个
                inputs_embeds=inputs_embeds,
                use_cache=use_cache)
        else:
            transformer_outputs = self.rwtranrsformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache)

        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1)

        # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
        # [prompt, answer, 0, 0, 0, 0] this is normal
        assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
        bs = values.size(0)
        seq_len = input_ids.shape[1]
        chosen_end_scores = []  # we use this name for consistency with the original forward function
        for i in range(bs):
            input_id = input_ids[i]
            value = values[i]

            c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
            # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
            c_ind = c_inds[0].item() + prompt_length if len(c_inds) > 0 else seq_len
            chosen_end_scores.append(value[c_ind - 1].item())
        return chosen_end_scores