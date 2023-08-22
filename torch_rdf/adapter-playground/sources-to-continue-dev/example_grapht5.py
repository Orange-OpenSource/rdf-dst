# Copyright 2023 Orange
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Software Name : knowledge-graph-dst
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Author: H. Andres Gonzalez

import torch
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration, PreTrainedModel, T5Model
from transformers.adapters import T5AdapterModel
from torch import nn


#t5 = T5ForConditionalGeneration.from_pretrained("t5-small")
t5 = T5AdapterModel.from_pretrained("t5-small")

t5.add_seq2seq_lm_head("tito")

print("TITO THE CAT")
#self.add_seq2seq_lm_head(task_name)
#self.set_active_adapters(task_name)
#self.train_adapter(task_name) # Freeze other model params.

