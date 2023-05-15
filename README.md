<!--
 * @Author: lihaitao
 * @Date: 2023-05-08 17:52:50
 * @LastEditors: Do not edit
 * @LastEditTime: 2023-05-09 14:34:49
 * @FilePath: /lht/ChatGLM_LoRA/README.md
-->

# ChatGLM_multi_gpu_zero_Tuning

实现mult_gpu + zero 微调ChatGLM, 目前已实现LoRA, Ptuning-v2, Freeze三种微调方式。


## Data Structure

data.py中使用InstrutionDataset和InstrutionCollator处理数据。

每条数据包含instruction,inpuy,answer,示例如下:

> {"instruction": "一名年龄在70岁的女性，出现了晕厥、不自主颤抖、情绪不稳等症状，请详细说明其手术治疗和术前准备。", "input": "", "output": "该病需要进行电极导线、脉冲发生器和永久心脏起搏器置入术，并需要使用镇静药物和局麻对病人进行手术治疗。术前准备包括1-3天的时间进行术前检查和生活方式的调整。"}

InstrutionDataset中的prefix只有ptuning时使用, InstrutionCollator中max_len和max_input_len用来控制输入长度

## Train

### LoRA

1.修改lora.sh中模型和数据路径

2.运行sh lora.sh
```bash
    CUDA_VISIBLE_DEVICES=${TOT_CUDA} deepspeed --master_port=$PORT --num_gpus=3 lora.py \
        --train_path ./instrution_data.json \
        --max_len 768 \
        --max_input_len 512 \
        --model_name_or_path ./chatGLM-6B \
        --tokenizer_name ./chatGLM-6B \
        --lora_rank 8 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 10 \
        --save_steps 900 \
        --learning_rate 1e-5 \
        --fp16 \
        --remove_unused_columns false \
        --logging_steps 50 \
        --output_dir /output \
        --deepspeed /ds_config.json \
```
单卡运行可以改为 num_gpus == 1

LoRA的参数如下,可根据实际情况调整:

```bash
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        lora_alpha=32,  
        target_modules=["query_key_value"],
        inference_mode=False,
        r=training_args.lora_rank,
        lora_dropout=0.1,
        bias="none",
        fan_in_fan_out = False
    )
```

### P-tuning-v2

根据ChatGLM-6B官方P—tuning代码修改。

1.修改ptuning.sh中模型和数据路径

2.运行sh ptuning.sh
```bash
    CUDA_VISIBLE_DEVICES=${TOT_CUDA} deepspeed --master_port=$PORT --num_gpus=2 finetune_ptuning.py \
        --train_path ./instrution_data.json \
        --max_len 768 \
        --max_input_len 512 \
        --model_name_or_path /chatGLM-6B \
        --tokenizer_name/chatGLM-6B \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 10 \
        --save_steps 2000 \
        --learning_rate 1e-5 \
        --fp16 \
        --logging_steps 50 \
        --pre_seq_len $PRE_SEQ_LEN \
        --output_dir /output \
        --deepspeed ds_config.json \
```

其中$PRE_SEQ_LEN是soft prompt的长度, 可以根据实际情况修改。

### Freeze

1.修改freeze.sh中模型和数据路径

2.运行sh freeze.sh

```bash
    CUDA_VISIBLE_DEVICES=${TOT_CUDA} deepspeed --master_port=$PORT --num_gpus=3 finetune_freeze.py \
        --train_path  \
        --max_len 768 \
        --max_input_len 512 \
        --model_name_or_path /chatGLM-6B \
        --tokenizer_name /chatGLM-6B \
        --lora_rank 8 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --num_train_epochs 10 \
        --save_steps 900 \
        --learning_rate 1e-5 \
        --fp16 \
        --remove_unused_columns false \
        --logging_steps 50 \
        --output_dir output_freeze \
        --deepspeed ds_config.json \
```

可通过以下代码修改可训练的层数:
```bash
    for name, param in model.named_parameters():
        if not any(nd in name for nd in ["layers.27", "layers.26", "layers.25", "layers.24", "layers.23"]):
            param.requires_grad = False
```

## Todo

- [ ] 增加模型并行和多卡inference

- [ ] 开源微调数据和微调模型


## Contact

If you find our work useful, please do not save your star!

If you have any questions, please email liht22@mails.tsinghua.edu.cn