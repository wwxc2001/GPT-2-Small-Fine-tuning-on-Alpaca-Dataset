from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--optim", type=str, default="adamw-torch")
    parser.add_argument("--optim_args", type=None, default="")
    parser.add_argument("--ga", default=1, type=int)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--bs", type=int, default=32)
    return parser.parse_args()


def main(args):    
    dataset = load_dataset("parquet", data_files="/opt/tiger/vtcl/optimize/dataset/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet")["train"]
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    tokenizer = AutoTokenizer.from_pretrained("/opt/tiger/vtcl/optimize/model/gpt2", padding_side="left", padding=True)
    tokenizer.pad_token = tokenizer.eos_token 

    def tokenize_function(example):
        encoding = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        encoding["labels"] = encoding["input_ids"].copy()
        return encoding

    def format_prompt(example):
        if example["input"]:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
        return {
            "prompt": prompt,
            "response": example["output"],
            "full_text": prompt + example["output"]
        }

    def tokenize_sft(example):
        # 分别 tokenize prompt 和 full_text
        prompt_ids = tokenizer(
            example["prompt"],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        full = tokenizer(
            example["full_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=512,
            padding="max_length",
        )

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]

        # 构建 labels：默认复制 input_ids
        labels = input_ids.copy()

        # 将 prompt 部分设为 -100
        labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

        # 将 padding 部分（attention_mask = 0）设为 -100
        labels = [
            label if mask == 1 else -100
            for label, mask in zip(labels, attention_mask)
        ]

        full["labels"] = labels
        return full



    # train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    # print(train_dataset[0])
    # eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)
    train_dataset = train_dataset.map(format_prompt)
    eval_dataset = eval_dataset.map(format_prompt)

    # Tokenize with label masking
    train_dataset = train_dataset.map(tokenize_sft, batched=False, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(tokenize_sft, batched=False, remove_columns=eval_dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained("/opt/tiger/vtcl/optimize/model/gpt2")

    training_args = TrainingArguments(
        output_dir=f"/opt/tiger/vtcl/optimize/trained_model/gpt2-{args.lr}_{int(args.bs*args.ga)}/{args.optim}{'' if args.optim_args is None else '_' + args.optim_args.replace('=', '-')}",
        per_device_train_batch_size=args.bs,
        num_train_epochs=5,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.ga,
        optim=args.optim,
        optim_args=args.optim_args,
        save_strategy="no",
        seed=42,
        eval_strategy="steps",
        eval_steps=0.02,
        fp16=True,
        report_to="tensorboard",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    args = get_args()
    main(args)