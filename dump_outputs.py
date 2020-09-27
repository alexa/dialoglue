import json
import os

hwu_command = """
CUDA_VISIBLE_DEVICES=0 python run.py \
        --train_data_path data_utils/dialoglue/hwu/train.csv \
        --val_data_path data_utils/dialoglue/hwu/val.csv \
        --test_data_path data_utils/dialoglue/hwu/test.csv \
        --token_vocab_path bert-base-uncased-vocab.txt \
        --output_dir {} \
        --train_batch_size 64 --dropout 0.1 --num_epochs 0 --learning_rate 6e-5 \
        --model_name_or_path bert-base-uncased --task intent --do_lowercase --max_seq_length 50 --mlm_pre --mlm_during --dump_outputs \
"""

banking_command = """
CUDA_VISIBLE_DEVICES=0 python run.py \
        --train_data_path data_utils/dialoglue/banking/train.csv \
        --val_data_path data_utils/dialoglue/banking/val.csv \
        --test_data_path data_utils/dialoglue/banking/test.csv \
        --token_vocab_path bert-base-uncased-vocab.txt \
        --output_dir {} \
        --train_batch_size 64 --dropout 0.1 --num_epochs 0 --learning_rate 6e-5 \
        --model_name_or_path bert-base-uncased --task intent --do_lowercase --max_seq_length 100 --mlm_pre --mlm_during --dump_outputs \
"""

clinc_command = """
CUDA_VISIBLE_DEVICES=0 python run.py \
        --train_data_path data_utils/dialoglue/clinc/train.csv \
        --val_data_path data_utils/dialoglue/clinc/val.csv \
        --test_data_path data_utils/dialoglue/clinc/test.csv \
        --token_vocab_path bert-base-uncased-vocab.txt \
        --output_dir {} \
        --train_batch_size 64 --dropout 0.1 --num_epochs 0 --learning_rate 6e-5 \
        --model_name_or_path bert-base-uncased --task intent --do_lowercase --max_seq_length 50 --mlm_pre --mlm_during --dump_outputs \
"""

restaurant8k_command = """
CUDA_VISIBLE_DEVICES=0 python run.py                  \
        --train_data_path data_utils/dialoglue/restaurant8k/train.json \
        --val_data_path data_utils/dialoglue/restaurant8k/val.json \
        --test_data_path data_utils/dialoglue/restaurant8k/test.json \
        --token_vocab_path bert-base-uncased-vocab.txt \
        --output_dir {} \
        --train_batch_size 64 --dropout 0.1 --num_epochs 0 --learning_rate 6e-5 \
        --model_name_or_path bert-base-uncased --task slot --do_lowercase --max_seq_length 50 --mlm_pre --mlm_during --dump_outputs \
"""

dstc8_command = """
CUDA_VISIBLE_DEVICES=0 python run.py                  \
        --train_data_path data_utils/dialoglue/dstc8_sgd/train.json \
        --val_data_path data_utils/dialoglue/dstc8_sgd/val.json \
        --test_data_path data_utils/dialoglue/dstc8_sgd/test.json \
        --token_vocab_path bert-base-uncased-vocab.txt \
        --output_dir {} \
        --train_batch_size 64 --dropout 0.1 --num_epochs 0 --learning_rate 6e-5 \
        --model_name_or_path bert-base-uncased --task slot --do_lowercase --max_seq_length 50 --mlm_pre --mlm_during --dump_outputs \
"""

top_command = """
CUDA_VISIBLE_DEVICES=0 python run.py                  \
        --train_data_path data_utils/dialoglue/top/train.txt \
        --val_data_path data_utils/dialoglue/top/eval.txt \
        --test_data_path data_utils/dialoglue/top/test.txt \
        --token_vocab_path bert-base-uncased-vocab.txt \
        --output_dir {} \
        --train_batch_size 64 --dropout 0.1 --num_epochs 0 --learning_rate 6e-5 \
        --model_name_or_path bert-base-uncased --task top --do_lowercase --max_seq_length 100 --mlm_pre --mlm_during --dump_outputs \
"""

multiwoz_command = """
cd trippy;

TASK="multiwoz21"
DATA_DIR="data/MULTIWOZ2.1"
OUT_DIR={0}

args_add="--do_eval --predict_type=test"

CUDA_VISIBLE_DEVICES=0 python3 run_dst.py \
        --task_name=${TASK} \
        --data_dir=${DATA_DIR} \
        --dataset_config=dataset_config/${TASK}.json \
        --model_type="bert" \
        --model_name_or_path="bert-base-uncased" \
        --do_lower_case \
        --learning_rate=1e-4 \
        --num_train_epochs=50 \
        --max_seq_length=180 \
        --per_gpu_train_batch_size=48 \
        --per_gpu_eval_batch_size=1 \
        --output_dir=${OUT_DIR} \
        --save_epochs=20 \
        --logging_steps=10 \
        --warmup_proportion=0.1 \
        --adam_epsilon=1e-6 \
        --label_value_repetitions \
        --swap_utterances \
        --append_history \
        --use_history_labels \
        --delexicalize_sys_utts \
        --class_aux_feats_inform \
        --class_aux_feats_ds \
        --seed 42 \
        --mlm_pre \
        --mlm_during \
        ${args_add} \
    2>&1 | tee ${OUT_DIR}/test.log

python3 metric_bert_dst.py \
        ${TASK} \
        dataset_config/${TASK}.json \
        "${OUT_DIR}/pred_res.test.json" \
        2>&1 | tee ${OUT_DIR}/eval_pred_test.log
"""


commands = [
    hwu_command,
    banking_command,
    clinc_command,
    restaurant8k_command,
    dstc8_command,
    top_command,
    multiwoz_command
]

checkpoints = [
    "checkpoints/hwu/",
    "checkpoints/banking/",
    "checkpoints/clinc/",
    "checkpoints/restaurant8k/",
    "checkpoints/dstc8/",
    "checkpoints/top/",
    "checkpoints/multiwoz/",
]

datasets = [
    "hwu",
    "banking",
    "clinc",
    "restaurant8k",
    "dstc8",
    "top",
    "multiwoz",
]

for cmd,ckpt in zip(commands, checkpoints):
    if "multiwoz" in ckpt:
        open("mwoz_command_temp.sh", "w+").write(cmd.replace("{0}", "../" + ckpt))
        cmd = "bash mwoz_command_temp.sh"      
    else:
        continue

    os.system(cmd.format(ckpt))

output_dict = {}
for dataset,ckpt in zip(datasets,checkpoints):
    if dataset == "multiwoz":
        data = json.load(open(ckpt + "pred_res.test.final.json"))
    else:
        data = json.load(open(ckpt + "outputs.json"))

    output_dict[dataset] = data

json.dump(output_dict, open("submission.json", "w+"))
