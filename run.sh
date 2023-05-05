# !/bin/bash -e
# https://keestalkstech.com/2022/03/named-arguments-in-a-bash-script/


# default epochs == 1 and devices == 0
epochs="${epochs:-1}"
devices="${devices:-0}"

programname=$0
function usage {
    echo ""
    echo "Runs DST experiments to generate DST as RDFs"
    echo ""
    echo "usage: $programname --testing_pipeline string --epochs integer --devices integer"
    echo ""
    echo "  --testing_pipeline string   yes or no"
    echo "                              (example: no)"
    echo "  --epochs integer            number of epochs to train for"
    echo "                              (example: 8)"
    echo "  --experiment integer        which experiment to run. 1, 2, or 3"
    echo "                              (example: 1)"
    echo "  --devices integer           which gpu to use"
    echo "                              (example: 0)"
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

if [[ -z $testing_pipeline ]]; then
    usage
    die "Missing parameter --testing_pipeline"
elif [[ -z $experiment ]]; then
    usage
    die "Missing parameter --experiment"
fi

# turning to lowercase
testing_pipeline="${testing_pipeline,,}"

if [[ $testing_pipeline != "yes" ]] && [[ $testing_pipeline != "no" ]]; then
    usage
    die "Incorrect parameter. Type either yes or no --testing_pipeline"
fi

if [[ $testing_pipeline == "yes" ]]; then
    CUDA_VISIBLE_DEVICES=$devices python main.py -epochs "$epochs" -d all -store yes -logger no -experiment "$experiment"
else
    CUDA_VISIBLE_DEVICES=$DEVICES python main.py -epochs 5 --batch 8 -d multiwoz -workers 6 -store no -experiment "$experiment"
fi
