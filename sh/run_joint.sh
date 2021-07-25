CUDA_VISIBLE_DEVICE=0
OUTPUT_DIR=./output
MODEL_PATH=./pretrained_model/chinese-bert-wwm
TRAIN_FILE=./data/tiny_train.json
VAL_FILE=./data/tiny_train.json
TEST_FILE=./data/tiny_train.json
LOGGING_DIR=$OUTPUT_DIR/runs
TRAIN_BS=1
EVAL_BS=8
LR=5e-5
EPOCHS=1
FP16=false
TASK=ner
LOGGING_STRATEGY=epoch
SAVE_STRATEGY=epoch
EVAL_STRATEGY=epoch
OVERWRITE_OUTPUT_DIR=true
SAVE_TOTAL_LIMIT=10
EVAL_ACC_STEPS=10
CACHE_DIR=./.cache
PREPRO_NUM_WORKERS=24
NUM_WORKERS=24
NO_CUDA=false

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICE python run_joint.py \
    --model_name_or_path $MODEL_PATH \
    --train_file $TRAIN_FILE \
    --validation_file $VAL_FILE \
    --test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size $TRAIN_BS \
    --per_device_eval_batch_size $EVAL_BS \
    --learning_rate $LR \
    --num_train_epochs $EPOCHS \
    --fp16 $FP16 \
    --task_name $TASK \
    --logging_strategy $LOGGING_STRATEGY \
    --save_strategy $SAVE_STRATEGY \
    --overwrite_output_dir $OVERWRITE_OUTPUT_DIR \
    --evaluation_strategy $EVAL_STRATEGY \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --eval_accumulation_steps $EVAL_ACC_STEPS \
    --dataloader_num_workers $NUM_WORKERS \
    --preprocessing_num_workers $PREPRO_NUM_WORKERS \
    --cache_dir $CACHE_DIR \
    --no_cuda $NO_CUDA