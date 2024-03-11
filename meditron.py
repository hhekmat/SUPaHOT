import os
import re
import sys
import json
import asyncio
import requests
import json
import asyncio
import aiofiles
import aiohttp
from preprocess import populate_global_resources, global_resource_dict
import backoff

MAX_ASYNC_TASKS = 250

@backoff.on_exception(backoff.expo, asyncio.TimeoutError, max_tries=8)
async def generate_meditron_response_async(session, prompt, task_prompt, model="meditron-7b"):
    SYSTEM_PROMPT = "You are a helpful medical assistant, users ask you questions pertaining to their health care information. You will help and be as concise and clear as possible."
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT + ' ' + task_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    async with session.post("http://35.229.176.120:8000/v1/chat/completions", json=data) as response:
        if response.status == 200:
            response_data = await response.json()
            if response_data['choices'] and response_data['choices'][0]['message']:
                return response_data['choices'][0]['message']['content'].strip()
            else:
                return "No response generated."
        else:
            print(f"Error during API call: {response.status}")
            return None
    
async def process_task_1():
    base_dir = 'queries'
    output_dir = 'task_1/output/meditron'
    finetune_dir = 'task_1/finetune/meditron'
    relevant_data_dir = 'all_resources'

    task_1_prompt = "Given a query and a resource from a patient's medical record, your job is to determine if the resource is relevant to providing an answer to the patient's query about their medical history. Respond only with 'True' if the resource could be relevant, or 'False' if the resource would not be helpful in providing the patient an answer to their question. Do not add any more detail, just answer with the words True or False." 

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:  # Single session
        for root, dirs, files in os.walk(base_dir):
            meditron_tasks = []
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    async with aiofiles.open(file_path, 'r') as f:
                        lines = await f.readlines()

                    query = lines[0].strip()
                    stripped_filename = re.sub(r'\d+', '', file.split('.')[0])
                    relevant_data_file = os.path.join(relevant_data_dir, f"{stripped_filename}resources.txt")

                    if os.path.exists(relevant_data_file):
                        finetune_data = []
                        meditron_tasks = []
                        resource_lines = []
                        relevant_resources = []
                        all_responses = []
                        async with aiofiles.open(relevant_data_file, 'r') as f:
                            async for resource_line in f:
                                resource = resource_line.strip()
                                prompt = f"Query: {query}, resource: {resource}"
                                print('prompt ', prompt)

                                meditron_tasks.append(generate_meditron_response_async(session, prompt, task_1_prompt))
                                resource_lines.append(resource)

                        all_responses = []
                        for i in range(0, len(meditron_tasks), MAX_ASYNC_TASKS):
                            batch_tasks = meditron_tasks[i:i+MAX_ASYNC_TASKS]
                            responses = await asyncio.gather(*batch_tasks)
                            all_responses.extend(responses)

                        for resource, meditron_response in zip(resource_lines, all_responses):
                            if meditron_response == "True":
                                relevant_resources.append(resource)
                            finetune_data.append({"query": query, "resource": resource, "label": meditron_response})

                        output_subdir = os.path.join(output_dir, os.path.relpath(root, base_dir))
                        os.makedirs(output_subdir, exist_ok=True)
                        output_file = os.path.join(output_subdir, file)

                        async with aiofiles.open(output_file, 'w') as f:
                            for resource in relevant_resources:
                                await f.write("%s\n" % resource)

                        finetune_subdir = os.path.join(finetune_dir, os.path.relpath(root, base_dir))
                        os.makedirs(finetune_subdir, exist_ok=True)
                        finetune_file = os.path.join(finetune_subdir, file.replace('.txt', '.jsonl'))

                        async with aiofiles.open(finetune_file, 'w') as f:
                            for item in finetune_data:
                                await f.write(json.dumps(item) + '\n')

                        print(f'Processed {file_path} -> {output_file}')
                        print(f'Finetuning data saved to {finetune_file}')

if __name__ == '__main__':
    # Accepting task number as a command-line argument
    if len(sys.argv) > 1:
        task = int(sys.argv[1])
        if task == 1:
            asyncio.run(process_task_1())
        # elif task == 3: call process_task_3()
        elif task == 2:
            populate_global_resources("./mock_patients")
            # process_task_2()
        elif task == 3: 
            print('entering task 3')
        else:
            print("Invalid task number. Please choose 1, 2, or 3.")
    else:
        print("Please provide a task number as a command-line argument.")

