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

    preds = [all_rdfs for pred in preds for all_rdfs in pred]
    labels = [all_rdfs for label in labels for all_rdfs in label]
    ids = [all_rdfs for i in ids for all_rdfs in i]

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
