from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, LongT5ForConditionalGeneration
import evaluate
import argparse

def main(model, size, dataset):
    data_for_project = load_dataset("rdfdial", "multiwoz")
    #random_dataset = load_dataset("datcha", "ATH-TV-resolution_dialogue")
    #random_dataset = load_dataset(dataset, 'all')
    gleu = evaluate.load("google_bleu")
    meteor = evaluate.load("meteor")

    if model == "flan-t5":
        loaded_model = T5ForConditionalGeneration.from_pretrained(f"google/flan-t5-{size}")
    elif model == "t5":
        loaded_model = T5ForConditionalGeneration.from_pretrained(f"t5-{size}")
    elif model == "long-t5-local":
        loaded_model = LongT5ForConditionalGeneration.from_pretrained(f"google/long-t5-local-{size}")
    elif model == "long-t5-tglobal":
        loaded_model = LongT5ForConditionalGeneration.from_pretrained(f"google/long-t5-tglobal-{size}")

    print(f"Model {model} successfully loaded")

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model", "--model", default='t5', type=str,
        choices=['t5', 'flan-t5', 'long-t5-local', 'long-t5-tglobal'],  # adapter model
        help="Select transformer"
    )

    parser.add_argument(
        "-model_size", "--model_size", default='base', type=str,
        choices=['small', 'base', 'large', 'xl'],
        help="Select size of transformer"
    )

    parser.add_argument(# need other key for this
        "-d", "--dataset", default='calor_dial', type=str,
        #choices=['sfxdial', 'multiwoz', 'dstc2', 'all', 'multiwoz-sim', 'camrest-sim'],
        help="Select data from options."
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = create_arg_parser()
    size = args.model_size
    model = args.model
    dataset = args.dataset
    main(model, size, dataset)
