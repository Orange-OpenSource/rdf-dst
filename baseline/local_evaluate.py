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
