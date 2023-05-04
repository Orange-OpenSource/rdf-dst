import argparse

def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch", default=4, type=int, help="Provide the number of batch"
    )  # 32?

    #parser.add_argument(
    #    "-d", "--data_dir", default='./sfx_rdf_data/', type=str,
    #    choices=['./sfx_rdf_data/', './multiwoz_rdf_data/', './dstc2_rdf_data/'],
    #    help="Select rdf data from options. Note that DATCHA is missing"
    #)

    parser.add_argument(
        "-d", "--dataset", default='sfxdial', type=str,
        choices=['sfxdial', 'multiwoz', 'dstc2', 'all'],
        help="Select rdf data from options. Note that DATCHA is missing"
    )

    parser.add_argument(
        "-epochs", "--epochs", default=1, type=int, help="Provide the number of epochs"
    )

    parser.add_argument(
        "-workers", "--num_workers", default=0, type=int, help="Provide the number of workers"
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
        "-s_len",
        "--source_length",
        default=512,
        type=int,
        help="define Max sequence length"
    )

    parser.add_argument(
        "-t_len",
        "--target_length",
        default=128,
        type=int,
        help="define Max sequence length"
    )

    parser.add_argument(
        "-store", "--store_output", default='no', type=str,
        choices=['yes', 'no'],
        help="Store output during inference"
    )

    args = parser.parse_args()
    return args
