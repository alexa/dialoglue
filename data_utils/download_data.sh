#!/usr/bin/env bash

download_intent_data() {
  cd intent_scripts
  bash get_all_data.sh ../dialoglue
  cd -

  python3 process_intent.py dialoglue/hwu
  python3 process_intent.py dialoglue/banking
  python3 process_intent.py dialoglue/clinc
  
  echo "Done downloading intent datasets"
}

download_slot_data() {
  git clone https://github.com/PolyAI-LDN/task-specific-datasets

  mv task-specific-datasets/span_extraction/restaurant8k/ dialoglue/restaurant8k
  mv task-specific-datasets/span_extraction/dstc8/ dialoglue/dstc8_sgd

  python3 process_slot.py dialoglue/restaurant8k
  python3 process_slot.py dialoglue/dstc8_sgd

  rm -rf task-specific-datasets
}

download_top_data() {
  curl -L "http://fb.me/semanticparsingdialog" --output sem.zip
  unzip sem.zip
  python3 process_top.py top-dataset-semantic-parsing/

  mkdir dialoglue/top
  cp top-dataset-semantic-parsing/train.txt dialoglue/top
  cp top-dataset-semantic-parsing/train_10.txt dialoglue/top
  cp top-dataset-semantic-parsing/eval.txt dialoglue/top
  cp top-dataset-semantic-parsing/test.txt dialoglue/top
  cp top-dataset-semantic-parsing/vocab.* dialoglue/top

  rm -rf top-dataset-semantic-parsing
  rm sem.zip
}

download_multiwoz_data() {
  git clone https://gitlab.cs.uni-duesseldorf.de/general/dsml/trippy-public.git
  cd trippy-public/data/MULTIWOZ2.1/
  gunzip *gz
  cd -
  mkdir dialoglue/multiwoz
  cp trippy-public/data/MULTIWOZ2.1/* dialoglue/multiwoz/
  rm -rf trippy-public

  python3 process_multiwoz.py
}


mkdir dialoglue
download_intent_data
download_slot_data
download_top_data
download_multiwoz_data
python3 merge_data.py
