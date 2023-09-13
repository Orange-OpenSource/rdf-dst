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
