import argparse

def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
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
        choices=['online', 'local'],
        help="Select to load data locally or from HF. When running from colab or jean zay, using local."
    )

    parser.add_argument(
        "-epochs", "--epochs", default=1, type=int, help="Provide the number of epochs"
    )

    parser.add_argument(
        "-workers", "--num_workers", default=0, type=int, help="Provide the number of workers"
    )

    parser.add_argument(
        "-experiment", "--experimental_setup", default=1, type=int,
        choices=[1, 2, 3],
        help="Select experimental setup.\n1: Context + states\n2: Context\n3: States"
    )

    parser.add_argument(
        "-subset", "--subsetting", default='no', type=str,
        choices=['yes', 'no'], help="Provide the number of epochs"
    )

    parser.add_argument(
        "-acc", "--accelerator", default='cuda', type=str,  # cuda or gpu?
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
        "-model", "--model", default='base', type=str,
        choices=['t5', 'flan-t5', 'long-t5-local', 'long-t5-tglobal'],  # adapter model
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