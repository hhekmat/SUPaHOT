import os
import sys
from openai import OpenAI
from process import global_resource_dict

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_oracle_response(query, relevant_data, task):
    # Customize the prompt based on the task
    if task == 1:
        prompt = f"Identify relevant resources from the following list based on the query: {query}. Resources: {relevant_data}"
    elif task == 2:
        prompt = f"Summarize the following resources into 1-2 sentences each: {relevant_data}"
    elif task == 3:
        prompt = f"Provide a concise answer to the query: {query}, based on the following summaries: {relevant_data}"
    else:
        return "Invalid task."

    # Using the new chat completions API format
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful, respectful and honest assistant. Answer as concisely as possible."},
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

def process_files(task):
    base_dir_mapping = {1: 'queries', 2: 'task_1/oracle', 3: 'task_2/oracle'}
    relevant_data_dir_mapping = {1: 'all_resources', 2: 'task_1/oracle', 3: 'task_2/oracle'}
    output_dir_mapping = {1: 'task_1', 2: 'task_2', 3: 'task_3'}

    base_dir = base_dir_mapping.get(task)
    output_dir = output_dir_mapping.get(task)
    relevant_data_dir = relevant_data_dir_mapping.get(task)

    if not base_dir or not output_dir:
        print("Invalid task number. Please choose 1, 2, or 3.")
        return

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                if task != 2:  # Task 1 and 3 include a query
                    query = lines[0].strip()
                else:  # Task 2 does not include a query, entire file content is relevant data
                    query = None

                # Construct the path to the relevant data file
                relevant_data_file = os.path.join(relevant_data_dir, file)
                if os.path.exists(relevant_data_file):
                    with open(relevant_data_file, 'r') as f:
                        relevant_data_lines = f.readlines()
                        if task == 2: 
                            relevant_data = '\n'.join([global_resource_dict[line.strip()] for line in relevant_data_lines])
                        else:
                            relevant_data = '\n'.join([line.strip() for line in relevant_data_lines])
                else:
                    relevant_data = "No relevant data found."


                oracle_response = generate_oracle_response(query, relevant_data, task)

                output_file = os.path.join(output_dir, file)

                os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

                with open(output_file, 'w') as f:
                    f.write(oracle_response)

                print(f'Processed {file_path} -> {output_file}')

if __name__ == '__main__':
    # Accepting task number as a command-line argument
    if len(sys.argv) > 1:
        task = int(sys.argv[1])
        process_files(task)
    else:
        print("Please provide a task number as a command-line argument.")
