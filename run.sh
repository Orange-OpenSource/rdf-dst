# !/bin/bash -e
# https://keestalkstech.com/2022/03/named-arguments-in-a-bash-script/


# default values
experiment="${experiment:-1}"
devices="${devices:-0}"
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
    echo "  --devices integer           which gpu to use"
    echo "                              (example and default val: 0)"
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


source ./dst-snake/bin/activate
if [[ $test == "yes" ]]; then
    CUDA_VISIBLE_DEVICES=$devices python main.py -epochs "$epochs" -d all -store yes -logger no -experiment "$experiment"
else
    CUDA_VISIBLE_DEVICES=$devices python main.py -epochs 5 --batch 8 -d multiwoz -workers 6 -store no -experiment "$experiment"
fi
