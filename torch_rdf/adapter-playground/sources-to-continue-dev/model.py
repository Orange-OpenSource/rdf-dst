# Copyright (c) 2023 Orange

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITEDTOTHE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Software Name : knowledge-graph-dst
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Author: H. Andres Gonzalez


# https://leoribeiro.github.io/
# CREDIT: https://github.com/AlibabaResearch/DAMO-ConvAI/blob/45ac3158281c5e5e86837b701a299b39974fe5f5/ssll/unifymodel/model.py#L3
# MORE SOURCES: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/45ac3158281c5e5e86837b701a299b39974fe5f5/graphix
import torch
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration, PreTrainedModel, T5Model
from transformers.adapters import T5AdapterModel
from torch import nn

class SSLLModel(T5AdapterModel):
    # def __init__(self, config, args=None):
    #     super(SSLLModel, self).__init__(config)
    #     self.args = args
        # self.model = T5AdapterModel(config)
        # self.model.transformer is T5Model
    def initialize(self, model_path, task_name, args=None):
        self = self.from_pretrained('t5-base', cache_dir=model_path)
        self.config.vocab_size = self.config.vocab_size + len(args.tasks) + 2 # Add task-specific special tokens. 2 for <ANS> and <QUES>
        self.transformer.resize_token_embeddings(self.config.vocab_size)
        # print('old config',self.config,flush=True)
        # print('decoder_start_token_id',self.config.decoder_start_token_id, flush=True)
        if args.model_aug:
            self.config.dropout_rate = args.dropout_rate
        return self

    def fresh(self, task_name, args=None, ema=False, initial=True): 
        # * Add task-specific adapters. 
        # self.add_adapter('adapter'+task_name)
        # self.add_seq2seq_lm_head('lm_head'+task_name)
        # self.train_adapter('adapter'+task_name)
        task_name = task_name.replace('.','_')
        if initial: 
            self.add_adapter(task_name)
            self.add_seq2seq_lm_head(task_name)
        self.set_active_adapters(task_name)
        self.train_adapter(task_name) # Freeze other model params.

        if not args.freeze_plm:
            self.freeze_model(False) # Finetuning all parameters. Default is True.
        # print('new config',self.config,flush=True)
        # print('decoder_start_token_id',self.config.decoder_start_token_id, flush=True)
        if ema:
            for param in self.parameters():
                param.detach_()
        return self
