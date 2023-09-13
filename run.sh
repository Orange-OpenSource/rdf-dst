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

DIR=./dst-snake

# default values, workers must be 1 with marcel... 6 with nadia?
#experiment="${experiment:-1}"
workers=5
#framework="baseline"
#model="t5"  # t5, flant-t5, long-t5-local, long-t5-tglobal

programname=$0
function usage {
    echo ""
    echo "Runs DST experiments to generate DST as RDFs"
    echo ""
    echo "usage: $programname --debug string --experiment integer --setup string --framework string --model string --size string"
    echo ""
    echo "  --setup string    	        training or evaluating"
    echo "                              (example: train, evaluate)"
    echo "  --framework string          using HF, torch, baseline or lightning"
    echo "                              (example: pl, torch, hf)"
    echo "  --debug string   		    yes or no"
    echo "                              (example: no)"
    echo "  --dataset string   		    multiwoz, dstc2, sfx"
    echo "                              (example: multiwoz)"
    echo "  --model string              t5, flan-t5, long-t5-local or long-t5-tglobal"
    echo "                              (example: long-t5-tglobal)"
    echo "  --size string               small, base, large. NO small version available for long models"
    echo "                              (example: small)"
    echo "  --experiment integer        which experiment to run. 1, 2, or 3"
    echo "                             (example and default val: 1)"
    echo "  --batch integer             select batch size 16, 8, or 4" 
    echo "                              (example : 4)" 
    echo "  --ig  string               select yes or no. Ignore inter states for rdf so yes. In baseline should be yes" 
    echo "                              (example : no)" 
    echo ""
}
function die {
    printf "Script failed: %s\n\n" "$1"
    exit 1
}


while [ $# -gt 0 ]; do
    if [[ $1 == "--help" ]]; then
        usage
        exit 0
    elif [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

if [[ -z $debug || -z $setup || -z $model || -z $size || -z $batch || -z $ig || -z $dataset ]]; then
	    usage
	        die "Missing parameter --debug or --setup or --model or --batch or --size or --ignore or --dataset"
fi

if [[ -z $framework ]]; then
    framework="torch"
fi

# turning to lowercase
debug="${debug,,}"

if [[ $debug != "yes" ]] && [[ $debug != "no" ]]; then
    usage
    die "Incorrect parameter. Type either yes or no --debug"
fi


handle_option(){
	case $1 in
		"baseline")
			cd baseline/
			;;

		"torch")
			cd torch_rdf/
			;;
	esac
}
# shell doesn't return variables per se, but if previously defined, the function overwrites the global var
handle_option "$framework"

if [[ $setup == "train" ]]; then
    script="model_train.py"

elif [[ $setup == "evaluate" ]]; then
    script="model_evaluate.py"
else
    usage
    die "Invalid value for setup parameter"
fi

eval_batch=$((batch * 2))
if [[ $debug == "yes" ]]; then
    python "$script" -epochs 2 -d multiwoz --batch 16 -store yes -logger no -experiment "$experiment" \
    -workers "$workers" -model "$model" -model_size "$size" -subset yes -device cuda -method online -peft lora
elif [[ $debug == "no" ]]; then
    if [[ $framework == "baseline" ]]; then
        CUDA_VISIBLE_DEVICES=1 python "$script" -epochs 5 --batch "$batch" -d "$dataset" -workers "$workers" -store yes \
        -experiment "$experiment" -model "$model" -incl "$ig" -model_size "$size" \
        -logger no -subset no -method offline -beam 5
    else
        python "$script" -epochs 5 --batch "$batch" -d "$dataset" -workers "$workers" \
        -store yes -experiment "$experiment" -model "$model" -ig "$ig" \
        -model_size "$size" -logger no -subset no -method offline -beam 5
        #CUDA_VISIBLE_DEVICES=0 python model_evaluate.py -epochs 5 --batch "$eval_batch" -d multiwoz -workers "$workers" -store yes -experiment "$experiment" -model "$model" -model_size "$size" -logger no -subset no -method offline -beam 5
    fi
else
    usage
    die "Invalid value for debug parameter"
fi

