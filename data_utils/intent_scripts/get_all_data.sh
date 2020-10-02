#!/usr/bin/env bash
# Original source: https://github.com/PolyAI-LDN/polyai-models
set -euf -o pipefail

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
DATASETS=("hwu" "clinc" "banking")
OUTPUT_DIR=$1
mkdir -p $OUTPUT_DIR

download_data () {
  dataset_path=${OUTPUT_DIR}/${1}

  echo "Downloading dataset $1 into $dataset_path"
  python "$SCRIPTPATH/get_$1_data.py" --data_dir $dataset_path
  echo "Dataset has been downloaded"

  echo "Creating train_10.csv, etc..."
  python "$SCRIPTPATH/subsample_from_train.py" --train_file "$dataset_path/train.csv" --n_per_class 10
  python "$SCRIPTPATH/subsample_from_train.py" --train_file "$dataset_path/train.csv" --n_per_class 30

  echo "Done!"
}

for i in "${!DATASETS[@]}";
do
  echo "Do you wish to download dataset ${DATASETS[$i]}?"
  select yn in "Yes" "No"; do
      case $yn in
          Yes ) download_data "${DATASETS[$i]}"; break;;
          No ) echo "OK, no problem!"; break;;
      esac
  done
done
