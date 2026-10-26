from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import os
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llmhgsyn.instructions import format_prompt, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract all-node multi-scale teacher states")
    parser.add_argument("--model", default="LLMHGSyn/models/Qwen2.5-7B-Instruct", help="Base LLM name or local path")
    parser.add_argument("--adapter", default="LLMHGSyn/outputs/teacher_lora/warm_start/fold_5",help="Fine-tuned LoRA adapter path")
    parser.add_argument("--instructions", default="LLMHGSyn/outputs/instructions/warm_start/fold_5/all.jsonl", help="all.jsonl from prepare_instructions.py")
    parser.add_argument("--output", default="LLMHGSyn/outputs/teacher/warm_start/fold_5.npz", help="Output .npz")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--layer-pooling",
        choices=["last", "last4_mean"],
        default="last4_mean",
        help="Hidden-layer pooling used for each prompt representation",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Physical GPU index",
    )
    parser.add_argument("--load-in-4bit", action="store_true",default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"Missing teacher extraction dependency: {exc}. Install requirements.txt")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"

    model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    device = next(model.parameters()).device

    records = read_jsonl(args.instructions)
    graph_ids = {str(record.get("graph_id", "")) for record in records}
    if len(graph_ids) != 1 or not next(iter(graph_ids)):
        raise ValueError("Instructions must contain one non-empty graph_id")
    graph_id = next(iter(graph_ids))
    num_nodes = max(int(record["center_node"]) for record in records) + 1
    num_scales = max(int(record["hop"]) for record in records) + 1
    extracted = {}
    with torch.inference_mode(): #关闭梯度计算
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size] #读取之前的prompt
            prompts = [format_prompt(record) for record in batch] #去除target
            encoded = [] #token列表
            for prompt in prompts:
                ids = tokenizer.encode(prompt, add_special_tokens=False) #转换为token id
                bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
                budget = args.max_length - len(bos)
                if len(ids) > budget:
                    head = max(1, int(budget * 0.7))
                    tail = budget - head
                    ids = ids[:head] + (ids[-tail:] if tail else [])
                encoded.append(bos + ids)
            batch_length = max(map(len, encoded))
            input_ids = [ids + [tokenizer.pad_token_id] * (batch_length - len(ids)) for ids in encoded] #补齐
            attention_mask = [[1] * len(ids) + [0] * (batch_length - len(ids)) for ids in encoded]
            inputs = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
            }
            outputs = model(**inputs, output_hidden_states=True, return_dict=True, num_logits_to_keep=1)
            if args.layer_pooling == "last4_mean":
                final_hidden = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0) #最后四层隐藏状态平均
            else:
                final_hidden = outputs.hidden_states[-1] #最后一层的隐藏层状态
            final_index = inputs["attention_mask"].sum(dim=1) - 1  #取最后一个真实token 因为Qwen是因果语言模型，最后一个真实的可以看到前面所有的
            batch_index = torch.arange(len(batch), device=device)
            vectors = final_hidden[batch_index, final_index].float().cpu().numpy() #取每条prompt的最终表征
            for record, vector in zip(batch, vectors): #按照节点和尺度保存表征
                key = (int(record["hop"]), int(record["center_node"]))
                extracted.setdefault(key, []).append(vector)

    missing = [
        (scale, node)
        for scale in range(num_scales)
        for node in range(num_nodes)
        if (scale, node) not in extracted
    ] #检查是否缺少节点或者尺度
    if missing:
        raise ValueError(f"Teacher extraction is incomplete; first missing entries: {missing[:5]}")
    teacher = np.stack(
        [
            np.stack(
                [np.stack(extracted[(scale, node)]).mean(axis=0) for node in range(num_nodes)]
            )
            for scale in range(num_scales)
        ]
    ).astype(np.float32)
    prompts_per_node_scale = np.asarray(
        [
            [len(extracted[(scale, node)]) for node in range(num_nodes)]
            for scale in range(num_scales)
        ],
        dtype=np.int64,
    ) #记录信息
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True) #保存
    np.savez_compressed(
        output,
        teacher_features=teacher,
        num_nodes=np.asarray(num_nodes),
        num_scales=np.asarray(num_scales),
        prompts_per_node_scale=prompts_per_node_scale,
        model=np.asarray(args.model),
        adapter=np.asarray(args.adapter or ""),
        layer_pooling=np.asarray(args.layer_pooling),
        graph_id=np.asarray(graph_id),
        split=np.asarray(str(records[0].get("split", ""))),
        fold=np.asarray(int(records[0].get("fold", -1))),
        node_names=np.asarray(
            [
                next(
                    str(record["node_name"])
                    for record in records
                    if int(record["center_node"]) == node and int(record["hop"]) == 0
                )
                for node in range(num_nodes)
            ]
        ),
    )
    print(json.dumps({"shape": list(teacher.shape), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
