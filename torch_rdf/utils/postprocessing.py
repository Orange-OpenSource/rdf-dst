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

import re

def clean_node(node):

# _:booking/0699e989  BEFORE ALL OF THIS FIND PATTERNS LIKE THIS IN A GIVEN NODE, REPLACE FORWARD AND CODE WITH PAD TOKEN
    node = node.strip()
    node = node.replace('_:', '')
    node = node.replace(',', '')  # removing commas to facilitate RDF creation and splitting
    node = node.replace(';', '')  # removing semicolons to facilitate RDF creation and splitting

    #node = node.replace(':', '')
    node = node.replace('USER', '') 
    node = node.replace('SYSTEM', '') 
    node = node.replace('STATE', '') 
    underscoreRegex = re.compile(r"_")
    node = underscoreRegex.sub(' ', node).lower().strip()
    mask = ""  # Replace with empty mask to make it closer to NL
    randompatternRegex = re.compile(r'\/[a-zA-Z0-9]+')
    return randompatternRegex.sub(mask, node)

def clean_rdf(rdf):
    rdf = rdf.split(';')
    # rdfs must be 3 elements. Removing patterns that break this rule
    if len(rdf) != 3:
        return None
    return ';'.join([clean_node(node) for node in rdf])


def clean_row(row):
    new_rows = []
    for rdfs in row:
        post_process_rdf = clean_rdf(rdfs)
        if post_process_rdf:
            new_rows.append(post_process_rdf)

    return list(frozenset(new_rows))

def postprocess_rdfs(decoded_batch):
    """
    returns several rdf triplets per batch
    """

    decoded_batch = [row.split(',') for row in decoded_batch]
    decoded_batch = map(clean_row, decoded_batch)
    return list(decoded_batch)
        
    #return [list(frozenset(clean_rdf(rdfs) for rdfs in row)) for row in decoded_batch]

