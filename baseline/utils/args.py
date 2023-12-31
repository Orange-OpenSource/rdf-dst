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

import argparse

def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()

    parser.add_argument("-peft", "--peft", default=None, type=str,
                        choices=['lora', 'adalora', 'prefix', 'ia3'], help="What peft method to use. If None then no peft")


    parser.add_argument("-incl", "--incl", default='yes', type=str,
                        choices=['yes', 'no'], help="Include intermediate states. Should be included in baseline to update belief states with sys responses")

    parser.add_argument("-beam", "--beam", default=2, type=int,
                        choices=[1, 2, 3, 4, 5], help="beam size, if beam == 1 then greedy")

    parser.add_argument(
        "-b", "--batch", default=4, type=int, help="Provide the number of batch"
    ) 

    parser.add_argument(
        "-d", "--dataset", default='sfxdial', type=str,
        choices=['sfxdial', 'multiwoz', 'dstc2', 'all', 'multiwoz-sim', 'camrest-sim'],
        help="Select rdf data from options. Note that DATCHA is missing"
    )

    parser.add_argument(
        "-method", "--method", default='online', type=str,
        choices=['online', 'offline'],
        help="Select to load data locally or from HF. When running from colab or jean zay, using offline."
    )

    parser.add_argument(
        "-epochs", "--epochs", default=1, type=int, help="Provide the number of epochs"
    )

    parser.add_argument(
        "-workers", "--num_workers", default=0, type=int, help="Provide the number of workers"
    )

    parser.add_argument(
        "-experiment", "--experimental_setup", default=1, type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="Select experimental setup.\n1: Context + states\n2: Context\n3: States"
    )

    parser.add_argument(
        "-subset", "--subsetting", default='no', type=str,
        choices=['yes', 'no'], help="Provide the number of epochs"
    )

    parser.add_argument(
        "-device", "--device", default='cuda', type=str,  # cuda or gpu?
        choices=['hpu', 'cpu', 'tpu', 'cuda', 'ipu', 'auto', 'mps'], help="Provide the number of epochs"
    )

    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass."
        )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1e-3,
        type=float,
        help="Provide the learning rate"
    )

    parser.add_argument(
        "-wd",
        "--weight_decay",
        default=0.0,
        type=float,
        help="Provide weight decay"
    )

    parser.add_argument(
        "-store", "--store_output", default='no', type=str,
        choices=['yes', 'no'],
        help="Store output during inference"
    )

    parser.add_argument(
        "-model", "--model", default='t5', type=str,
        choices=['t5'],  # adapter model
        help="Select transformer"
    )

    parser.add_argument(
        "-model_size", "--model_size", default='base', type=str,
        choices=['small', 'base', 'large', 'xl'],
        help="Select size of transformer"
    )

    parser.add_argument(
        "-logger", "--logger", default='no', type=str,
        choices=['yes', 'no'],
        help="Logging with w&b and tensorboard"
    )

    args = parser.parse_args()
    return args
