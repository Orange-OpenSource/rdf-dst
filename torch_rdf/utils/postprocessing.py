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

