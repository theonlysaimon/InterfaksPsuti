pip instal torch==1.4.0
pip3 install transformers==3.5.0

git clone https://github.com/sberbank-ai/ru-gpts

mkdir models/

wget -O train.txt https://www.dropbox.com/s/oa3v9c7g9bp40xw/train.txt?dl=0
wget -O valid.txt https://www.dropbox.com/s/mworl3ld6r3bg62/valid.txt?dl=0

export PYTHONPATH=${PYTHONPATH}:/ru-gpts/
CUDA_VISIBLE_DEVICES=0 python ru-gpts/pretrain_transformers.py \
    --output_dir=models/essays \
    --model_type=gpt2 \
    --model_name_or_path=sberbank-ai/rugpt3small_based_on_gpt2 \
    --do_train \
    --train_data_file=train.txt \
    --do_eval \
    --eval_data_file=valid.txt \
    --per_gpu_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --block_size 2048 \
    --overwrite_output_dir

python ru-gpts/generate_transformers.py \
    --model_type=gpt2 \
    --model_name_or_path=models/essays \
    --k=5 \
    --p=0.95 \
    --length=500 \
    --repetition_penalty=5

