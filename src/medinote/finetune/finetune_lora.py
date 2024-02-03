
from dataclasses import dataclass, field
import logging
import pathlib
import typing
import os
import gc

import pandas
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
from transformers import Trainer, BitsAndBytesConfig, deepspeed
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import Dataset

import torch

from fastchat.train.train import (
    DataArguments,
    ModelArguments,
)

from fastchat.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)
from fastchat.train.train_lora import get_peft_state_maybe_zero_3
from medinote.finetune.overwrite import peft_initialization

current_path = os.path.dirname(__file__)

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)
logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
transformers.logging.set_verbosity_info()

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: typing.Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    flash_attn: bool = False
    flash_rotary: bool = False
    # fused_dense: bool = False
    # low_cpu_mem_usage: bool = False
    samples_start_index : int = -1
    samples_end_index : int = -1
    target_model_path : str = ""
    deepspeed: str = None


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules_str: str = "q_proj,v_proj"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

def train(body: dict = None):
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )       
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses() if body is None else parser.parse_dict(body)
           
    if training_args.flash_attn:
        replace_llama_attn_with_flash_attn()

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logger.warning(
                "FSDP and ZeRO3 are both currently incompatible with QLoRA."
            )

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    kwargs = {}
    if training_args.flash_attn:
        kwargs["flash_attn"] = training_args.flash_attn
    if training_args.flash_rotary:
        kwargs["flash_rotary"] = training_args.flash_rotary
    if training_args.low_cpu_mem_usage:
        kwargs["low_cpu_mem_usage"] = training_args.low_cpu_mem_usage
    if training_args.fused_dense:
        kwargs["fused_dense"] = training_args.fused_dense
    if training_args.device_map:
        device_map = training_args.device_map
    if training_args.trust_remote_code:
        kwargs["trust_remote_code"] = training_args.trust_remote_code
        
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        ) if lora_args.q_lora else None,
        **kwargs,
    )
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules_str.split(","),
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
        if not ddp and torch.cuda.device_count() > 1:
            # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
            model.is_parallelizable = True
            model.model_parallel = True

    model = get_peft_model(model, lora_config)
    if training_args.flash_attn:
        for name, module in model.named_modules():
            if "norm" in name:
                module = module.to(compute_dtype)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    module = module.to(compute_dtype)
    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    def tokenize(sample):
        tokenized_text =  tokenizer(sample["text"], padding=True, truncation=True, max_length=512)
        return tokenized_text
    logger.info(f"Loading the dataset from {data_args.data_path}")
    
    with open(data_args.data_path, "r") as f:
        data_df = pandas.read_json(f, lines=True if data_args.data_path.endswith(".jsonl") else False)
    data = Dataset.from_pandas(data_df)
    if training_args.samples_start_index > -1 and training_args.samples_start_index < training_args.samples_end_index:
        logger.info(f"Size of samples : {data.shape[0]}")
        if training_args.samples_end_index > data.shape[0]:
            logger.info(f"End index is greater than size of samples, setting end index to {data.shape[0]}")
            training_args.samples_end_index = data.shape[0]
        data = data.select(range(training_args.samples_start_index, training_args.samples_end_index))
        training_args.output_dir = f"{training_args.output_dir}_{training_args.samples_start_index}_{training_args.samples_end_index}"
        
    tokenized_data = data.map(tokenize, batched=True, desc="Tokenizing data", remove_columns=data.column_names)
    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    logger.info("Loading peft")
    peft_initialization()


    trainer = Trainer(
        model=model,
        train_dataset=tokenized_data,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False
    logger.info("Training the model ....")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resuming from checkpoint **************************")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        # use deepspeed engine internal function to gather state dict
        # state_dict_zero3 contains whole parameters of base and lora adapters
        # we will not extract lora parameters since peft save_pretrained will do that
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
        # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        # in other mode we use original code from fastchat team, to make sure our change is minimum
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )

    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)

    if body:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return "Training complete."
    

if __name__ == "__main__":
    train()
