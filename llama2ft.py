import os
import re
import sys
import json
import backoff
import requests
import time
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from preprocess import global_resource_dict, populate_global_resources

api_token = os.getenv('TOGETHER_API_KEY')

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-hf"
headers = {f"Authorization": {api_token}}

'''# Initialize the LLaMA model and tokenizer for the chat model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
generate = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # Adjust 'device' as per your setup'''

def giveup_condition(details):
    """Condition to give up retrying, for example when status code is not 429."""
    exc = details['value']
    # Assuming 'exc' is the raised requests.exceptions.HTTPError
    if exc.response.status_code != 429:
        return True
    return False

@backoff.on_exception(
    backoff.expo,
    requests.exceptions.HTTPError,  # Retrying on HTTP errors
    giveup=giveup_condition,  # Custom giveup condition
    max_time=300,  # Total max time to retry
    jitter=backoff.full_jitter,  # Adding jitter to spread out retry attempts
)


def generate_llama_response(user_prompt, task_prompt, model):
    endpoint = 'https://api.together.xyz/v1/chat/completions'
    headers = {
        "Authorization": "Bearer f4a39e6ede5141b23e5ba2febeb4428589a55db917e72cbe5a4234e3a16a669b"
    }

    try:
        input_text = f"[INST] You are a helpful medical assistant. Users ask you questions about their health care information. You will help and be as concise and clear as possible.\n\n{task_prompt}\n\n{user_prompt}\n[/INST]"
        payload = {
            "model": model,
            "max_tokens": 2048,
            "prompt": input_text,
            "temperature": 0.01,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1,
            "stop": ["[/INST]", "</s>"],
            "repetitive_penalty": 1
        }
        
        response = query(endpoint, headers, payload)
        if response and 'choices' in response and response['choices']:
            return response['choices'][0]['message']['content'].strip()
        else:
            return "No response generated."
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

def query(endpoint, headers, payload):
    response = requests.post(endpoint, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait_time = int(retry_after) if retry_after else 10  # Default to 30 seconds if header is missing
            print(f"Hit rate limit, retrying after {wait_time} seconds...")
            time.sleep(wait_time)  # Sleep for the time specified in the Retry-After header
            return query(endpoint, headers, payload)  # Recursive retry after waiting
        else:
            print(f"Error during API call: {e}")
            raise
    return response.json()

'''def query(endpoint, headers, payload):
    response = requests.post(endpoint, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()'''


# Adapting the file processing functions for task 1 and task 2
def process_task_1(model):
    base_dir = 'queries'
    output_dir = 'task_1/output/llama_ft'
    relevant_data_dir = 'all_resources'

    task_1_prompt = "Given a query and a resource from a patient's medical record, your job is to determine if the resource could potentially be relevant to providing an answer to the patient's query about their medical history. Respond only with 'True' if the resource may be relevant, or 'False' if the resource would not be helpful at all in providing the patient an answer to their question." 
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                query = lines[0].strip()
                resources = lines[1:]
                stripped_filename = re.sub(r'\d+', '', file.split('.')[0])
                relevant_data_file = os.path.join(relevant_data_dir, f"{stripped_filename}resources.txt")

                if os.path.exists(relevant_data_file):
                    relevant_resources = []
                    c = 0
                    for resource in resources:
                        c += 1
                        print(c)
                        resource = resource.strip()
                        prompt = f"Query: {query}, resource: {resource}"

                        llama_response = generate_llama_response(prompt, task_1_prompt, model)
                        print(llama_response)
                        if llama_response.find("True") != -1:
                            print('true')
                            relevant_resources.append(resource)

                    output_subdir = os.path.join(output_dir, os.path.relpath(root, base_dir))
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file = os.path.join(output_subdir, file)

                    with open(output_file, 'w') as f:
                        f.writelines("%s\n" % resource for resource in relevant_resources)

                    print(f'Processed {file_path} -> {output_file}')

def process_task_2(model):
    base_dir = 'task_1/output/oracle'
    output_dir = 'task_2/output/llama_ft'
    task_2_prompt = "Given an excerpt of a JSON object corresponding to a resource from a patient's FHIR medical records, your job is to provide a brief (1 to 2 sentence) natural language summary of the JSON. Don't explicitly mention that it's a JSON."

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                process_file(file_path, root, file, base_dir, output_dir, task_2_prompt, model)


def process_file(file_path, root, file, base_dir, output_dir, task_2_prompt, model):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if len(lines) == 0:
        prewritten_response = "No relevant resources were found for this query."
        process_empty_file(root, file, base_dir, output_dir, prewritten_response)
    else:
        for line in lines:
            process_line(line, root, file, base_dir, output_dir, task_2_prompt, model)


def process_empty_file(root, file, base_dir, output_dir, prewritten_response):
    rel_path = os.path.relpath(root, base_dir)
    output_subdir = os.path.join(output_dir, rel_path)
    os.makedirs(output_subdir, exist_ok=True)
    output_file = os.path.join(output_subdir, file)

    with open(output_file, 'a') as f_txt:
        f_txt.write(f"{prewritten_response}\n")

def process_line(line, root, file, base_dir, output_dir, task_2_prompt, model):
    resource_label = line.strip()
    large_resource = global_resource_dict.get(resource_label, {})
    large_resource_str = json.dumps(large_resource)
    summary = generate_llama_response("JSON: " + large_resource_str, task_2_prompt, model)
    summary = ' '.join(summary.split())
    print(summary)

    rel_path = os.path.relpath(root, base_dir)
    output_subdir = os.path.join(output_dir, rel_path)
    os.makedirs(output_subdir, exist_ok=True)
    output_file = os.path.join(output_subdir, file)

    with open(output_file, 'a') as f_txt:
        f_txt.write(f"{summary}\n")

def process_task_3(model):
    query_dir = 'queries'
    summary_dir = 'task_2/output/oracle'
    output_dir = 'task_3/output/llama_ft'
    
    task_3_prompt = "You will be given a query from a patient who is inquiring about their medical records and a list of summaries (seperated by commas) of medical resources from the patient's medical record. Answer the patient's query using relevant information from the summaries."

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

                    combined_summaries = ", ".join(summaries).strip()
                    prompt = f"Query:'{query}' Summaries: {combined_summaries}"

                    answer = generate_llama_response(prompt, task_3_prompt, model)
                    answer = ' '.join(answer.split())

                    # Prepare output paths
                    rel_path = os.path.relpath(root, query_dir)
                    output_subdir = os.path.join(output_dir, rel_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_file = os.path.join(output_subdir, file)

                    # Write the answer in text format
                    with open(output_file, 'w') as f_txt:
                        f_txt.write(f"{answer}\n")

                    print(f'Processed {query_file_path} -> {output_file}')

if __name__ == '__main__':
    # Your command-line interface logic for executing different tasks
    if len(sys.argv) > 1:
        task = int(sys.argv[1])
        if task == 1:
            process_task_1("hahekmat777@gmail.com/llama-2-7b-post_task_1_ft-2024-03-14-11-29-51")
        elif task == 2:
            # Ensure any global resources are populated if necessary
            populate_global_resources("./mock_patients")
            process_task_2("hahekmat777@gmail.com/llama-2-7b-post_task_2_ft-2024-03-14-12-05-08")
        elif task == 3: 
            process_task_3("hahekmat777@gmail.com/llama-2-7b-post_task_3_ft-2024-03-14-12-06-32")
        else:
            print("Invalid task number. Please choose 1, 2, or 3.")
    else:
        print("Please provide a task number as a command-line argument.")
