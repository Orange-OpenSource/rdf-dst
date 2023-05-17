# !/bin/bash -e
# https://keestalkstech.com/2022/03/named-arguments-in-a-bash-script/


# default values
experiment="${experiment:-2}"
epochs=1

programname=$0
function usage {
    echo ""
    echo "Runs DST experiments to generate DST as RDFs"
    echo ""
    echo "usage: $programname --test string --experiment integer --devices integer"
    echo ""
    echo "  --test string   		yes or no"
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

if [[ -z $test ]]; then
    usage
    die "Missing parameter --test"
fi

# turning to lowercase
test="${test,,}"

if [[ $test != "yes" ]] && [[ $test != "no" ]]; then
    usage
    die "Incorrect parameter. Type either yes or no --test"
fi


echo "Using manual data loading"

python3 -m venv dst-snake
source ./dst-snake/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python old_orange_certs.py

if [[ $test == "yes" ]]; then
    python main.py -epochs 2 -d multiwoz -store yes -logger no -experiment "$experiment" -workers 6 -model small
else
    python main.py -epochs 5 --batch 4 -d multiwoz -workers 6 -store yes -experiment "$experiment" -model small -logger yes
fi
