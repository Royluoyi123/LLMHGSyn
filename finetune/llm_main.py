from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import os
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmhgsyn.instructions import format_prompt, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune the node-level teacher LLM")
    parser.add_argument("--model", default="LLMHGSyn/models/Qwen2.5-7B-Instruct", help="Hugging Face model name or local path")
    parser.add_argument("--instruction-dir", default="LLMHGSyn/outputs_DrugCombDB/instructions/warm_start/fold_1")
    parser.add_argument("--output-dir", default="LLMHGSyn/outputs_DrugCombDB/teacher_lora/warm_start/fold_1")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA module names",
    )
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--bf16", action="store_true",default=True)
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU编号,例如0或1",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class InstructionDataset:
    def __init__(self, records: List[Dict[str, object]], tokenizer, max_length: int):
        if max_length < 64:
            raise ValueError("max_length must be at least 64")
        self.examples = []
        eos = tokenizer.eos_token or ""
        bos_id = tokenizer.bos_token_id
        for record in records:
            prompt_ids = tokenizer.encode(format_prompt(record), add_special_tokens=False)
            target_ids = tokenizer.encode(str(record["target"]) + eos, add_special_tokens=False)
            bos = [bos_id] if bos_id is not None else []
            # Always preserve the response target and both ends of a long graph prompt.
            target_ids = target_ids[: max_length - len(bos) - 1]
            prompt_budget = max_length - len(bos) - len(target_ids)
            if len(prompt_ids) > prompt_budget:
                head = max(1, int(prompt_budget * 0.7))
                tail = prompt_budget - head
                prompt_ids = prompt_ids[:head] + (prompt_ids[-tail:] if tail else [])
            input_ids = bos + prompt_ids + target_ids
            prompt_length = len(bos) + len(prompt_ids)
            labels = [-100] * min(prompt_length, len(input_ids)) + input_ids[prompt_length:]
            self.examples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


class CausalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch

        max_length = max(len(item["input_ids"]) for item in features)
        input_ids, labels, attention_mask = [], [], []
        for item in features:
            padding = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            labels.append(item["labels"] + [-100] * padding)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM, #加载Qwen因果语言模型
            AutoTokenizer, #将文本转换为Token
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(f"Missing fine-tuning dependency: {exc}. Install requirements.txt")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)  #加载
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token #设置填充
    tokenizer.padding_side = "right" #右侧填充

    model_kwargs = {"torch_dtype": torch.bfloat16 if args.bf16 else "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs) #加载Qwen模型
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model) #为4-bit LoRA训练做准备
    model.config.use_cache = False
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    ) #定义LoRA
    model = get_peft_model(model, lora_config) #LoRA安装到Qwen
    model.print_trainable_parameters()

    instruction_dir = Path(args.instruction_dir) #找到制定数据目录
    train_data = InstructionDataset(read_jsonl(instruction_dir / "train.jsonl"), tokenizer, args.max_length) #读取数据


    lengths = [len(example["input_ids"]) for example in train_data.examples]
    lengths_sorted = sorted(lengths)

    print("训练样本数:", len(lengths))
    print("最短token数:", min(lengths))
    print("平均token数:", sum(lengths) / len(lengths))
    print("P95 token数:", lengths_sorted[int(len(lengths_sorted) * 0.95)])
    print("最长token数:", max(lengths))
    print("达到max_length的样本数:", sum(x == args.max_length for x in lengths))

    training_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs, #完整的训练集会训练几遍
        per_device_train_batch_size=args.batch_size, #1 每次处理几条prompt
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=args.bf16,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    ) #训练参数

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=train_data,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
