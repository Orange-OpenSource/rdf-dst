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

from penman import load as load_, Graph, Triple
from penman import loads as loads_
from penman import encode as encode_
from penman import decode
from penman.model import Model
from penman.models.noop import NoOpModel
from penman.models import amr

op_model = Model()
noop_model = NoOpModel()
amr_model = amr.model
DEFAULT = op_model

def _get_model(dereify):
    if dereify is None:
        return DEFAULT


    elif dereify:
        return op_model

    else:
        return noop_model

def _remove_wiki(graph):
    metadata = graph.metadata
    triples = []
    for t in graph.triples:
        v1, rel, v2 = t
        if rel == ':wiki':
            t = Triple(v1, rel, '+')
        triples.append(t)
    graph = Graph(triples)
    graph.metadata = metadata
    return graph

def load(source, dereify=None, remove_wiki=False):
    model = _get_model(dereify)
    out = load_(source=source, model=model)
    if remove_wiki:
        for i in range(len(out)):
            out[i] = _remove_wiki(out[i])
    return out

def loads(string, dereify=None, remove_wiki=False):
    model = _get_model(dereify)
    out = loads_(string=string, model=model)
    if remove_wiki:
        for i in range(len(out)):
            out[i] = _remove_wiki(out[i])
    return out

def encode(g, top=None, indent=-1, compact=False):
    model = amr_model
    return encode_(g=g, top=top, indent=indent, compact=compact, model=model)

def pm_decode(g):
    return decode(g)
