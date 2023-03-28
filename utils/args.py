import argparse

def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch", default=6, type=int, help="Provide the number of batch"
    )  # 32?

    parser.add_argument(
        "-d", "--data_dir", default='../sfx_rdf_data/', type=str,
        choices=['../sfx_rdf_data/', '../multiwoz_rdf_data/', '../dstc2_rdf_data/'],
        help="Select rdf data from options. Note that DATCHA is missing"
    )  # 32?

    parser.add_argument(
        "-epochs", "--epochs", default=1, type=int, help="Provide the number of epochs"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1e-3,
        type=float,
        help="Provide the learning rate"
    )
    parser.add_argument(
        "-l",
        "--seq_length",
        default=512,
        type=int,
        help="define Max sequence length"
    )


    args = parser.parse_args()
    return args
