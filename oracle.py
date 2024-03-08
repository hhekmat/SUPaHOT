import os
import re
import sys
import json
from openai import OpenAI

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_oracle_response(prompt):
    
    # Using the new chat completions API format for a single resource check
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Your job is to determine if a given resource from a patient's medical data file is relevant to answering a specific query about their health data. You will be given a query and a resource, and you will respond with 'True' or 'False' to indicate if the resource is relevant to the query."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256,
        temperature=0.01
    )

    # Extracting and returning the response
    if response.choices and response.choices[0].message:
        return response.choices[0].message.content.strip()
    else:
        return "No response generated."

def process_task_1():
    base_dir = 'queries'
    output_dir = 'task_1/output/oracle'
    finetune_dir = 'task_1/finetune/oracle'
    relevant_data_dir = 'all_resources'

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                query = lines[0].strip()
                stripped_filename = re.sub(r'\d+', '', file.split('.')[0])
                relevant_data_file = os.path.join(relevant_data_dir, f"{stripped_filename}resources.txt")

                if os.path.exists(relevant_data_file):
                    finetune_data = []
                    with open(relevant_data_file, 'r') as f:
                        relevant_resources = []
                        for resource_line in f:
                            resource = resource_line.strip()
                            prompt = f"Assume a patient has asked you the question '{query}', would the following resource from this patient's medical file be relevant in providing an answer to this query? Respond with 'True' or 'False'. The resource is {resource}"
                            oracle_response = generate_oracle_response(prompt)
                            if oracle_response == "True":
                                relevant_resources.append(resource)
                            finetune_data.append({"query": query, "resource": resource, "label": oracle_response})

                    output_subdir = os.path.join(output_dir, os.path.relpath(root, base_dir))
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file = os.path.join(output_subdir, file)

                    with open(output_file, 'w') as f:
                        f.writelines("%s\n" % resource for resource in relevant_resources)

                    finetune_subdir = os.path.join(finetune_dir, os.path.relpath(root, base_dir))
                    os.makedirs(finetune_subdir, exist_ok=True)
                    finetune_file = os.path.join(finetune_subdir, file.replace('.txt', '.jsonl'))

                    with open(finetune_file, 'w') as f:
                        for item in finetune_data:
                            f.write(json.dumps(item) + '\n')

                    print(f'Processed {file_path} -> {output_file}')
                    print(f'Finetuning data saved to {finetune_file}')

def process_task_2():
    base_dir = 'queries'
    output_dir = 'task_1/oracle/'
    relevant_data_dir = 'all_resources'

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                query = lines[0].strip()
                stripped_filename = re.sub(r'\d+', '', file.split('.')[0])
                relevant_data_file = os.path.join(relevant_data_dir, f"{stripped_filename}resources.txt")

                if os.path.exists(relevant_data_file):
                    with open(relevant_data_file, 'r') as f:
                        relevant_resources = []
                        for resource_line in f:
                            resource = resource_line.strip()
                            prompt = f"For the query: '{query}', is the following resource relevant? Respond with 'Y' or 'N': {resource}"
                            oracle_response = generate_oracle_response(prompt)
                            if oracle_response == "Y":
                                relevant_resources.append(resource)

                    output_subdir = os.path.join(output_dir, os.path.relpath(root, base_dir))
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file = os.path.join(output_subdir, file)

                    with open(output_file, 'w') as f:
                        f.writelines("%s\n" % resource for resource in relevant_resources)

                    print(f'Processed {file_path} -> {output_file}')

# Add similar functions for process_task_2() and process_task_3() as needed

if __name__ == '__main__':
    # Accepting task number as a command-line argument
    if len(sys.argv) > 1:
        task = int(sys.argv[1])
        if task == 1:
            process_task_1()
        # elif task == 2: call process_task_2()
        # elif task == 3: call process_task_3()
        else:
            print("Invalid task number. Please choose 1, 2, or 3.")
    else:
        print("Please provide a task number as a command-line argument.")
