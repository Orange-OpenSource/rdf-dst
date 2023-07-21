# !/bin/bash -e
# https://keestalkstech.com/2022/03/named-arguments-in-a-bash-script/

DIR=./dst-snake

# default values, workers must be 1 with marcel... 6 with nadia?
experiment="${experiment:-1}"
workers=5
framework="torch"
model="t5"  # t5, flant-t5, long-t5-local, long-t5-tglobal

programname=$0
function usage {
    echo ""
    echo "Runs DST experiments to generate DST as RDFs"
    echo ""
    echo "usage: $programname --debug string --experiment integer --devices integer"
    echo ""
    echo "  --setup          	        training or evaluating"
    echo "                              (example: train, evaluate)"
    echo "  --framework string          using HF, torch, or lightning"
    echo "                              (example: pl, torch, hf)"
    echo "  --debug string   		 yes or no"
    echo "                              (example: no)"
    echo "  --experiment integer        which experiment to run. 1, 2, or 3" echo "                              (example and default val: 1)"
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

if [[ -z $debug || -z $setup ]]; then
    usage
    die "Missing parameter --debug or --setup"
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
		"pl")
			cd lightning_rdf/
			;;
		"hf")
			cd hf_rdf/
			;;
		 
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

if [[ $debug == "yes" ]]; then
    python "$script" -epochs 2 -d multiwoz --batch 4 -store yes -logger no -experiment "$experiment" -workers "$workers" -model "$model" -model_size small -subset yes -device cuda -method online -peft yes
elif [[ $debug == "no" ]]; then
    python "$script" -epochs 5 --batch 16 -d multiwoz -workers "$workers" -store yes -experiment "$experiment" -model "$model" -model_size base -logger yes -subset no
else
    usage
    die "Invalid value for debug parameter"
fi

