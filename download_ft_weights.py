import os
import sys
import together

task_list = [1, 2, 3]
for task_num in task_list:
    output_path = f'./ft_model/post_task_{str(task_num)}/model.tar.zst'
    if task_num == 1:
        id = 'ft-9550ed16-7cbe-498d-9b21-4b86967a9ad0'
    elif task_num == 2:
        id = 'ft-61c8a738-9ee6-46af-99b0-d7aac3a69cd2'
    elif task_num == 3:
        id = 'ft-6859f5c5-2e2a-4454-95be-643d72059832'
    together.Finetune.download(
        fine_tune_id=id,
        output = output_path
    )