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

