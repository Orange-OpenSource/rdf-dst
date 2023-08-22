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
from torch_geometric.data import Data, Batch, DataLoader

def get_pytorch_graph_(embeddings, graphs):

    # print('embeddings.size()', embeddings.size())
    # print('len graphs', len(graphs))
    # exit()

    list_geometric_data = []

    # for each b in batch
    for idx, emb in enumerate(embeddings):
        #print(idx)
        edges_index = graphs[idx][0]
        edges_types = graphs[idx][1]
        #print(edges_index)

        data = Data(x=emb, edge_index=edges_index, y=edges_types)
        list_geometric_data.append(data)

    #print('len list', len(list_geometric_data))
    bdl = Batch.from_data_list(list_geometric_data)

    bdl = bdl.to('cuda:' + str(torch.cuda.current_device()))

    return bdl


def get_pytorch_graph(embeddings, graphs):

    # print('embeddings.size()', embeddings.size())
    # print('len graphs', len(graphs))
    # exit()

    list_geometric_data = [Data(x=emb, edge_index=graphs[idx][0], y=graphs[idx][1]) for idx, emb in enumerate(embeddings)]

    #print('len list', len(list_geometric_data))
    bdl = Batch.from_data_list(list_geometric_data)
    bdl = bdl.to('cuda:' + str(torch.cuda.current_device()))

    return bdl
