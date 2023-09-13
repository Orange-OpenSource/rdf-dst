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


# pip install python-gitlab
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from graph_collator import GraphCollator


tokenizer = AutoTokenizer.from_pretrained("t5-small")
collator = GraphCollator(tokenizer, "multiwoz")

data = load_dataset("rdfdial", "multiwoz")
data = concatenate_datasets([data['train'], data['validation'], data['test']])

data = data.map(collator, batched=True, remove_columns=['states'], load_from_cache_file=False)
print("TITO")
raise SystemExit

# all states flat? idk, double check
#print(data['states'][8])
state = data['states'][8]
turns = data['turns'][8]
print(len(state))
raise SystemExit
#print(data['turns'][0])
speaker_order = set()
for t in data['turns']:
    order = ''
    for s in t:
        order += (s['speaker'] + '-')
    order = order[:-1]

    speaker_order.add(order)
for x in speaker_order:
    print(x[-6:])
raise SystemExit
print()
print()
print(data['graph_states'][8])
edges = collator.edges
nodes_1 = collator.nodes_1
nodes_2 = collator.nodes_2
print(len(nodes_1))
print(len(edges))
print(len(nodes_2))
#sys_triples = [triple[:2] for s in data['states'] for triple in s]
#sys_triples = [trip for rdf in sys_triples for trip in rdf]
#all_trips = set()
#for trip in sys_triples:
#    all_trips.add('|||'.join(trip[:2]))
#print(all_trips)
print(edges)
# sys inquired, sys offered, sys canthelp, sys greeted, sys dismissed
#print(nodes_1)

tokenizer_vocab = set(tokenizer.vocab.keys())
raise SystemExit
states = data['states']
flat_states = [state for t in states for state in t]
flat_rdf = [triple for rdf in flat_states for triple in rdf]
rdf_vocab = set(word for triple in flat_rdf for word in triple)
tokenizer_vocab = set(tokenizer.vocab.keys())
print(len(rdf_vocab))
print(len(tokenizer_vocab))
common_tokens = rdf_vocab & tokenizer_vocab
print(len(common_tokens))
sub_tokens = rdf_vocab - tokenizer_vocab
print(len(sub_tokens))
sent = tokenizer.tokenize("tito is a cat")
print(sent)
counter = []
for word in rdf_vocab:
    res = tokenizer.tokenize(word)
    if '<unk>' in res:
        counter.append(1)
print(len(counter))
#https://www.youtube.com/watch?v=p_PcQB9KBxY
# the conversion... https://github.com/UKPLab/StructAdapt/blob/master/convert_graph_tokenizer.py
