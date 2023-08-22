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

from utils.metric_tools import DSTMetrics
import pandas as pd
import argparse
import os

def prepare_data(path, file):
    df = pd.read_csv(os.path.join(path, file))
    #outputs = df.to_dict('records')  # kinda funky when evaluating

    preds = [eval(pred) for pred in df['preds'].array]
    labels = [eval(label) for label in df['labels'].array]
    ids = [eval(id) for id in df['ids'].array]

    #preds = [all_rdfs for pred in preds for all_rdfs in pred]
    #labels = [all_rdfs for label in labels for all_rdfs in label]
    #ids = [all_rdfs for i in ids for all_rdfs in i]

    return {"preds": preds, "labels": labels, "ids": ids}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-path",
        "--path",
        default='.',
        type=str,
        help="Provide path where csv can be found"
    )

    parser.add_argument(
        "-file",
        "--file",
        default='outputs.csv',
        type=str,
        help="Provide csv file"
    )

    args = parser.parse_args()
    outputs = prepare_data(args.path, args.file)
    dst_metrics = DSTMetrics()
    scores = dst_metrics(outputs, from_file=True)
    print(scores)

if __name__ == '__main__':
    main()
