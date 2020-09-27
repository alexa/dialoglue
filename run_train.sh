CUDA_VISIBLE_DEVICES=0 python run.py                  \
--train_data_path data_utils/dialoglue/top/train.txt \
--val_data_path data_utils/dialoglue/top/eval.txt \
--test_data_path data_utils/dialoglue/top/test.txt \
--token_vocab_path bert-base-uncased-vocab.txt \
--output_dir checkpoints/top/ \
--train_batch_size 64 --dropout 0.1 --num_epochs 0 --learning_rate 6e-5 \
--model_name_or_path bert-base-uncased --task top --do_lowercase --max_seq_length 100 --mlm_pre --mlm_during --dump_outputs \

