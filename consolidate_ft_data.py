import os
import json
import random

def task_1_ft_dataset(inp, outp):
    for i in range(len(inp)):
        if i == 0:
            true_lim = 100
            false_lim = 300
        else:
            true_lim = 20
            false_lim = 60
        true_count = 0
        false_count = 0
        true_lines = []
        false_lines = []

        with open(outp[i], 'a') as outfile:
            for root, dirs, files in os.walk(inp[i]):
                for file in files:
                    if file.endswith('.jsonl'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as infile:
                            for line in infile:
                                if '"label": "True"' in line:
                                    true_count += 1
                                    if true_count <= true_lim:
                                        true_lines.append(line)
                                else:
                                    false_count += 1
                                    if false_count <= false_lim:
                                        false_lines.append(line)

            # Shuffle the lines
            all_lines = true_lines + false_lines
            random.shuffle(all_lines)

            # Write to the output file
            for line in all_lines:
                outfile.write(line)

def create_ft_datasets(inp, outp):
    for i in range(len(inp)):
        with open(outp[i], 'a') as outfile:
            for root, dirs, files in os.walk(inp[i]):
                for file in files:
                    if file.endswith('.jsonl'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as infile:
                            for line in infile:
                                outfile.write(line)
        

if __name__ == '__main__':
    inp = ['./task_1/finetune/oracle/train', './task_1/finetune/oracle/validation']
    outp = ['./ft_datasets/task_1_train.jsonl', './ft_datasets/task_1_val.jsonl']
    input_dirs = ['./task_2/finetune/oracle/train', './task_2/finetune/oracle/validation', './task_3/finetune/oracle/train', './task_3/finetune/oracle/validation']
    output_files = ['./ft_datasets/task_2_train.jsonl', './ft_datasets/task_2_val.jsonl', './ft_datasets/task_3_train.jsonl', './ft_datasets/task_3_val.jsonl']
    task_1_ft_dataset(inp, outp)
    create_ft_datasets(input_dirs, output_files)