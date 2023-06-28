# !/bin/bash -e
# https://keestalkstech.com/2022/03/named-arguments-in-a-bash-script/

DIR=./dst-snake

# default values, workers must be 1 with marcel... 6 with nadia?
experiment="${experiment:-3}"
workers=6
framework="torch"
script="empty"
model="t5"  # t5, flant-t5, long-t5-local, long-t5-tglobal

programname=$0
function usage {
    echo ""
    echo "Runs DST experiments to generate DST as RDFs"
    echo ""
    echo "usage: $programname --debug string --experiment integer --devices integer"
    echo ""
    echo "  --framework string          using HF, torch, or lightning"
    echo "                              (example: pl, torch, hf)"
    echo "  --debug string   		 yes or no"
    echo "                              (example: no)"
    echo "  --experiment integer        which experiment to run. 1, 2, or 3"
    echo "                              (example and default val: 1)"
    echo ""
}
function die {
    printf "Script failed: %s\n\n" "$1"
    exit 1
}


# NO NEED FOR VIRTUAL ENV WITH A CONTAINER!
#if [ -d "$DIR" ];
#then
#    echo "$DIR directory exists."
#else
#    echo "$DIR directory does not exist. Setting up virtual environment..."
#    ./setup.sh
#fi
#
#if [ ! -d "$DIR" ]; then
#    die "Virtual environment was not properly setup"
#fi
#
#source ./dst-snake/bin/activate

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

if [[ -z $debug ]]; then
    usage
    die "Missing parameter --debug"
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
			script="pl_main.py"
			#script="assess_marcel_pl.py"
			;;
		"hf")
			cd hf_rdf/
			script="hf_main.py"
			;;
		 
		"baseline")
			cd baseline/
			script="main.py"
			;;

		"torch")
			cd torch_rdf/
			script="main.py"
			#script="assess_marcel_torch.py"
			;;
	esac
}
# shell doesn't return variables per se, but if previously defined, the function overwrites the global var
handle_option "$framework"

if [[ $debug == "yes" ]]; then
    python "$script" -epochs 3 -d multiwoz -store yes -logger yes -experiment "$experiment" -workers "$workers" -model "$model" -model_size small -subset yes -acc cpu -method online
else
    python "$script" -epochs 5 --batch 8 -d multiwoz -workers "$workers" -store yes -experiment "$experiment" -model "$model" -model_size base -logger yes -subset no
fi
