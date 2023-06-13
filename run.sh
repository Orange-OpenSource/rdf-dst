# !/bin/bash -e
# https://keestalkstech.com/2022/03/named-arguments-in-a-bash-script/

DIR=./dst-snake

# default values, workers must be 1 with marcel... 6 with nadia?
experiment="${experiment:-1}"
workers=1
framework="torch"
script="empty"

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
    framework="pl"
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
			script="pl_main.py"
			#script="assess_marcel_pl.py"
			;;
		"hf")
			script="hf_main.py"
			;;
		 
		"torch")
			script="torch_main.py"
			#script="assess_marcel_torch.py"
			;;
	esac
}
# shell doesn't return variables per se, but if previously defined, the function overwrites the global var
handle_option "$framework"

if [[ $debug == "yes" ]]; then
    python "$script" -epochs 3 -d multiwoz -store yes -logger no -experiment "$experiment" -workers "$workers" -model small -subset yes -acc cpu
else
    python "$script" -epochs 5 --batch 16 -d multiwoz -workers "$workers" -store yes -experiment "$experiment" -model base -logger yes -subset no
fi
