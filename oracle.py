import os
import re
import sys
import json
from preprocess import global_resource_dict
from openai import OpenAI

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_oracle_response(prompt):
    
    # Using the new chat completions API format for a single resource check
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant, users ask you questions pertaining to their health care information. You will help and be as concise and clear as possible."},
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
    base_dir = 'task_1/output/oracle'
    output_dir = 'task_2/output/oracle'
    finetune_dir = 'task_2/finetune/oracle'

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    resource_label = line.strip()
                    large_resource = global_resource_dict.get(resource_label, {})
                    large_resource_str = json.dumps(large_resource)
                    summary = generate_oracle_response('Generate a 1-2 sentence summary for this JSON file: \n ' + large_resource_str)

                    # Prepare output paths
                    rel_path = os.path.relpath(root, base_dir)
                    output_subdir = os.path.join(output_dir, rel_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file = os.path.join(output_subdir, file)

                    finetune_subdir = os.path.join(finetune_dir, rel_path)
                    os.makedirs(finetune_subdir, exist_ok=True)
                    finetune_file = os.path.join(finetune_subdir, file.replace('.txt', '.jsonl'))

                    # Write summaries in text format
                    with open(output_file, 'a') as f_txt:
                        f_txt.write(f"{summary}\n")

                    # Write summaries in JSONLines format
                    with open(finetune_file, 'a') as f_jsonl:
                        f_jsonl.write(json.dumps({"resource_label": resource_label, "summary": summary}) + '\n')

                    print(f'Summarized {resource_label} -> {output_file}')
                    print(f'Summary data saved to {finetune_file}')

def process_task_3():
    query_dir = 'queries'
    summary_dir = 'task_2/output/oracle'
    output_dir = 'task_3/output/oracle'
    finetune_dir = 'task_3/finetune/oracle'

    for root, dirs, files in os.walk(query_dir):
        for file in files:
            if file.endswith('.txt'):
                query_file_path = os.path.join(root, file)
                with open(query_file_path, 'r') as f:
                    query = f.readline().strip()

                summary_file_path = os.path.join(summary_dir, os.path.relpath(root, query_dir), file)

                if os.path.exists(summary_file_path):
                    with open(summary_file_path, 'r') as f:
                        summaries = f.readlines()

                    combined_summaries = " ".join(summaries).strip()
                    prompt = f"Given the query '{query}' and the following summaries: {combined_summaries}, provide an answer based on this information."

                    answer = generate_oracle_response(prompt)

                    # Prepare output paths
                    rel_path = os.path.relpath(root, query_dir)
                    output_subdir = os.path.join(output_dir, rel_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file = os.path.join(output_subdir, file)

                    finetune_subdir = os.path.join(finetune_dir, rel_path)
                    os.makedirs(finetune_subdir, exist_ok=True)
                    finetune_file = os.path.join(finetune_subdir, file.replace('.txt', '.jsonl'))

                    # Write the answer in text format
                    with open(output_file, 'w') as f_txt:
                        f_txt.write(f"{answer}\n")

                    # Write the answer in JSONLines format
                    with open(finetune_file, 'w') as f_jsonl:
                        f_jsonl.write(json.dumps({"query": query, "answer": answer}) + '\n')

                    print(f'Processed {query_file_path} -> {output_file}')
                    print(f'Answer data saved to {finetune_file}')


if __name__ == '__main__':
    # Accepting task number as a command-line argument
    if len(sys.argv) > 1:
        task = int(sys.argv[1])
        if task == 1:
            process_task_1()
        elif task == 2:
            process_task_2()
        elif task == 3: 
            process_task_3()
        else:
            print("Invalid task number. Please choose 1, 2, or 3.")
    else:
        print("Please provide a task number as a command-line argument.")
