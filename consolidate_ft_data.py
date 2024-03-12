import os
import json

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
    input_dirs = ['./task_2/finetune/oracle/train', './task_2/finetune/oracle/validation', './task_3/finetune/oracle/train', './task_3/finetune/oracle/validation']
    output_files = ['./ft_datasets/task_2_train.jsonl', './ft_datasets/task_2_val.jsonl', './ft_datasets/task_3_train.jsonl', './ft_datasets/task_3_val.jsonl']
    create_ft_datasets(input_dirs, output_files)