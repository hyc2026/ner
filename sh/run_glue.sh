export TASK_NAME=cola
TRAIN_FILE=./data/tiny_train.json
VAL_FILE=./data/tiny_train.json
TEST_FILE=./data/tiny_train.json

python run_glue.py \
  --model_name_or_path ./pretrained_model/chinese-bert-wwm \
  --train_file $TRAIN_FILE \
  --validation_file $VAL_FILE \
  --test_file $TEST_FILE \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./output \
  --overwrite_output_dir true