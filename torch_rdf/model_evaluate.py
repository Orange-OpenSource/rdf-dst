from dotenv import load_dotenv
load_dotenv()  # load keys and especially w and biases to see visualizations. Looking in curr dir

import re
import os
import glob
# longt5 needs special module to avoid errors
from transformers import AutoTokenizer, T5ForConditionalGeneration, LongT5ForConditionalGeneration
from utils.data_loader import DialogueRDFData
from utils.args import create_arg_parser
from utils.metric_tools import DSTMetrics
from evaluator import MyEvaluation
from utils.predata_collate import PreDataCollator
from peft import PeftModel, PeftConfig

import logging

logging.basicConfig(level=logging.INFO)
SEED = 42  # for replication purposes

def preprocessing(collator, dataset, num_workers, batch_size, method):

    data = DialogueRDFData(collator, num_workers=num_workers,
                           dataset=dataset,
                           batch_size=batch_size,
                           inference_time=True)
    data.load_hf_data(method)
    dataloaders = data.create_loaders(subsetting=subsetting)
    global vocabulary
    vocabulary = data.vocabulary

    return {'test': dataloaders['test']}

def evaluating(model, tokenizer, test_dataloader, device, 
             target_len, dst_metrics, path):


    logging.info("Inference stage")


    my_evaluation = MyEvaluation(model, tokenizer, device, target_len, dst_metrics, path=path, is_peft=is_peft, vocabulary=vocabulary)
    my_evaluation(test_dataloader, validation=False, verbose=True)
    print(my_evaluation.results)

def load_model(file_path):

    ckpt_path = find_version_num(file_path)
    #ckpt_path = '../results/models/tb_logs/flan-t5_experiment_1/version_0/checkpoints/best_dst_ckpt/'

    if is_peft and 'long' not in file_path:
        peft_model_id = ckpt_path
        config = PeftConfig.from_pretrained(peft_model_id)
        model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, peft_model_id)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path) 
    else:
        if 'long' not in file_path:
            model = T5ForConditionalGeneration.from_pretrained(ckpt_path)
        else:
            model = LongT5ForConditionalGeneration.from_pretrained(ckpt_path)

        tokenizer = AutoTokenizer.from_pretrained(ckpt_path) 

    store_path = os.path.dirname(ckpt_path)
    return {"model": model, "tokenizer": tokenizer, "store_path": store_path}


def find_version_num(path):

    dirs = [d for d in os.listdir(path) if 'checkpoints' in os.listdir(os.path.join(path, d))]
    assert dirs, "No version has any checkpoints. Did you train the model?"
    newest_version = max(map(regex_match, dirs))
    parent_dir = os.path.join(path, f"version_{newest_version}", "checkpoints")
    pattern = os.path.join(parent_dir, "best_dst_ckpt")
    checkpoints = [dir_path for dir_path in glob.glob(pattern) if os.path.isdir(dir_path)]
    return max(checkpoints, key=os.path.getctime)

def regex_match(dir_name):
    versionRegex = re.compile(r"^version_(\d+)$")
    res_match = versionRegex.search(dir_name)
    if res_match:
        return int(res_match.group(1))
    else:
        return -1

def main():

    global subsetting
    global store
    global is_peft

    args = create_arg_parser()
    models = {'t5': 't5', 'flan-t5': 'google/flan-t5', 'long-t5-local': 'google/long-t5-local', 'long-t5-tglobal': 'google/long-t5-tglobal'}
    dataset = args.dataset
    num_workers = args.num_workers
    batch_size = args.batch
    method = args.method
    device = args.device
    model_name = models[args.model]

    bool_4_args = {"no": False, "yes": True}

    subsetting = bool_4_args[args.subsetting]
    store = bool_4_args[args.store_output]
    is_peft = bool_4_args[args.peft]


    length_exp_setup = {1: {"source_len": 1024, "target_len": 1024, "setup": "context and states"},
                        2: {"source_len": 512,  "target_len": 1024, "setup": "only context"},
                        3: {"source_len": 768,  "target_len": 1024, "setup": "only states"}}
    # TO DEBUG
    #length_exp_setup = {1: {"source_len": 256, "target_len": 256, "setup": "context and states"},
    #                    2: {"source_len": 256,  "target_len": 256, "setup": "only context"},
    #                    3: {"source_len": 256,  "target_len": 256, "setup": "only states"}}

    experimental_setup = args.experimental_setup

    source_len = length_exp_setup[experimental_setup]["source_len"]
    target_len = length_exp_setup[experimental_setup]["target_len"]
    model_checkpoint_name = f"{model_name}_{args.model_size}_experiment_{experimental_setup}"

    if os.getenv('DPR_JOB'):
        path = os.path.join("/userstorage/", os.getenv('DPR_JOB'))
    else:
        path = "."

    model_checkpoint_name = model_checkpoint_name.replace('google/', '')
    file_path = os.path.join(path, 'tb_logs', model_checkpoint_name)

    loaded_config = load_model(file_path)
    tokenizer = loaded_config["tokenizer"]
    model = loaded_config["model"]
    store_path = loaded_config["store_path"]


    cut_context = True if ((model_name[:2] == 't5') and (experimental_setup == 1)) else False
    collator = PreDataCollator(tokenizer, source_len, target_len, experimental_setup, cut_context=cut_context, inference_time=True)
    dataloaders = preprocessing(collator, dataset, num_workers, batch_size, method)



    dst_metrics = DSTMetrics()  # this is loading the metrics now so we don't have to do this again

    logging.info(f"Outputs will be stored in\n{store_path}")
    evaluating(model, tokenizer, dataloaders['test'], device, 
               target_len, dst_metrics, path=store_path)

if __name__ == '__main__':
    main()
