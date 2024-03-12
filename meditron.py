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

MAX_ASYNC_TASKS = 100

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
    # handling error prompts more elegantly because there are a lot of errors :P
    try: 
        async with session.post("http://35.201.169.217:8000/v1/chat/completions", json=data) as response:
            if response.status == 200:
                response_data = await response.json()
                if response_data['choices'] and response_data['choices'][0]['message']:
                    return response_data['choices'][0]['message']['content'].strip()
                else:
                    return "No response generated."
            else:
                print(f"Error during API call: {response.status}")
                return None
    except asyncio.TimeoutError:
        print(f"TimeoutError for prompt: {prompt}")
        return "TimeoutError"
    except Exception as e:
        print(f"Unexpected error: {e} for prompt: {prompt}")
        return "Error"
    
async def process_task_1():
    base_dir = 'queries'
    output_dir = 'task_1/output/meditron'
    finetune_dir = 'task_1/finetune/meditron'

    task_1_prompt = ("Given a query and a resource from a patient's medical record, your job is to determine if the resource could potentially be relevant to providing an answer to the patient's query about their medical history. You must respond only with 'True' if the resource may be relevant, or 'False' if the resource would not be helpful at all in providing the patient an answer to their question. Do not provide any additional detail or try to answer the original users query. Only answer True or False. Here is the query and respective resource:")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:  # Single session
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    async with aiofiles.open(file_path, 'r') as f:
                        lines = await f.readlines()

                    query = lines[0].strip()
                    resources = lines[1:]  # Treat subsequent lines as resources
                    finetune_data = []
                    meditron_tasks = []
                    resource_lines = []

                    for resource in resources:
                        resource = resource.strip()
                        prompt = f"This is the query: {query}, This is the resource: {resource}. Return True if the resource could be relevant to the query or False if it is certainly not related to the query. Return True or False."
                        print('prompt ', prompt)

                        meditron_tasks.append(generate_meditron_response_async(session, prompt, task_1_prompt))
                        resource_lines.append(resource)

                    all_responses = []
                    for i in range(0, len(meditron_tasks), MAX_ASYNC_TASKS):
                        batch_tasks = meditron_tasks[i:i+MAX_ASYNC_TASKS]
                        responses = await asyncio.gather(*batch_tasks)
                        all_responses.extend(responses)
                    print('all responses is ', all_responses)

                    for resource, meditron_response in zip(resource_lines, all_responses):
                        print('meditron response here was', meditron_response)
                        if meditron_response == "True":
                            finetune_data.append({"query": query, "resource": resource, "label": meditron_response})

                    output_subdir = os.path.join(output_dir, os.path.relpath(root, base_dir))
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file = os.path.join(output_subdir, file)

                    # Append relevant resources directly to the original query file
                    async with aiofiles.open(output_file, 'w') as f:
                        await f.write(query + '\n')
                        for data in finetune_data:
                            await f.write(f"{data['resource']}\n")

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

