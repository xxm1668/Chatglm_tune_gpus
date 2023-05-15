import os
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from data import InstrutionDataset, InstrutionCollator
from arguments import ModelArguments, DataTrainingArguments, FinetuneArguments as TrainingArguments

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


tokenizer = AutoTokenizer.from_pretrained('./chatGLM-6B', trust_remote_code=True)


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def main():
    writer = SummaryWriter()

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, trust_remote_code=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False
    )

    # setup peft
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        target_modules=["query_key_value"],
        inference_mode=False,
        r=training_args.lora_rank,
        lora_dropout=0.1,
        bias="none",
        fan_in_fan_out=False
    )
    model = get_peft_model(model, peft_config)
    model = model.half()
    model.print_trainable_parameters()

    ## data
    train_data = InstrutionDataset(
        data_path=data_args.train_path)

    data_collator = InstrutionCollator(
        tokenizer=tokenizer,
        max_len=training_args.max_len,
        max_input_len=training_args.max_input_len
    )

    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )

    trainer.train()
    writer.close()

    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
