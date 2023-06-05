# !/bin/bash -e
# https://keestalkstech.com/2022/03/named-arguments-in-a-bash-script/


source ./dst-snake/bin/activate
# default values
experiment="${experiment:-3}"
epochs=1

programname=$0
function usage {
    echo ""
    echo "Runs DST experiments to generate DST as RDFs"
    echo ""
    echo "usage: $programname --debug string --experiment integer --devices integer"
    echo ""
    echo "  --debug string   		    yes or no"
    echo "                              (example: no)"
    echo "  --experiment integer        which experiment to run. 1, 2, or 3"
    echo "                              (example and default val: 1)"
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

if [[ -z $debug ]]; then
    usage
    die "Missing parameter --debug"
fi

# turning to lowercase
debug="${debug,,}"

if [[ $debug != "yes" ]] && [[ $debug != "no" ]]; then
    usage
    die "Incorrect parameter. Type either yes or no --debug"
fi


echo "Using manual data loading"

if [[ $debug == "yes" ]]; then
    python main.py -epochs 3 -d multiwoz -store yes -logger no -experiment "$experiment" -workers 6 -model small -subset yes -acc gpu
else
    python main.py -epochs 5 --batch 8 -d multiwoz -workers 6 -store yes -experiment "$experiment" -model base -logger yes -subset no
fi
