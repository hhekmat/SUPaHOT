import os
import sys
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

def load_input_output_pairs(base_dir, system_prompt):
    input_output_pairs = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        input_prompt = system_prompt
                        for field in inp_fields:
                            inp_example = data.get(field)
                            if isinstance(inp_example, list):
                                for summary in inp_example:
                                    input_prompt += ', ' + summary
                            else:
                                input_prompt += inp_example
                        output_text = data.get(outp_field)
                        input_output_pairs.append((input_prompt, output_text))
    return input_output_pairs

def fine_tune_model(train_data, val_data, model_name, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_dir='./logs',
        logging_steps=100,
        eval_steps=500,
        save_steps=500,
        evaluation_strategy="steps",
        save_total_limit=5,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        dataloader_num_workers=4,
        logging_first_step=True,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)

    print("Training completed!")

if __name__ == '__main__':
    # Accepting task number as a command-line argument
    if len(sys.argv) > 1:
        task = int(sys.argv[1])
        model_name = 'epfl-llm/meditron-7b'
        if task == 1:
            train_data_dir = './ft_datasets/task_1_train.jsonl'
            val_data_dir = './ft_datasets/task_1_val.jsonl'
            output_dir = './ft_model/post_task_1'
            system_prompt = ''
            inp_fields = ['query', 'resource']
            outp_field = 'label'
        elif task == 2:
            train_data_dir = './ft_datasets/task_2_train.jsonl'
            val_data_dir = './ft_datasets/task_2_val.jsonl'
            output_dir = './ft_model/post_task_2'
            system_prompt = ''
            inp_fields = ['resource_label']
            outp_field = 'summary'
        elif task == 3: 
            train_data_dir = './ft_datasets/task_3_train.jsonl'
            val_data_dir = './ft_datasets/task_3_val.jsonl'
            output_dir = './ft_model/post_task_3'
            system_prompt = ''
            inp_fields = ['query', 'resource_summaries']
            outp_field = 'answer'
        else:
            print("Invalid task number. Please choose 1, 2, or 3.")

        train_data = load_input_output_pairs(train_data_dir, system_prompt, inp_fields, outp_field)
        val_data = load_input_output_pairs(val_data_dir, system_prompt, inp_fields, outp_field)

        fine_tune_model(train_data, val_data, model_name, output_dir)
    else:
        print("Please provide a task number as a command-line argument.")
