import os
import requests
import json

def process_file(file_path, output_dir):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Assuming the first line is the query and the rest is relevant data
    query = lines[0].strip()
    relevant_data = ' '.join([line.strip() for line in lines[1:]])

    data = {
        "model": "meditron-7b",
        "messages": [
            {"role": "user", "content": query},
            {"role": "system", "content": f"Based on the following patient data: {relevant_data}"}
        ]
    }

    response = requests.post("http://35.194.221.232:8000/v1/chat/completions", json=data)
    
    if response.status_code == 200:
        response_data = response.json()
        message_content = response_data['choices'][0]['message']['content']

        # Constructing output file path
        base_filename = os.path.basename(file_path)
        output_file_path = os.path.join(output_dir, base_filename)

        with open(output_file_path, 'w') as f:
            f.write(message_content)

        print(f'Processed {file_path} -> {output_file_path}')
    else:
        print(f'Error processing {file_path}: {response.status_code}')

def process_directory(input_dir='queries/test', output_dir='generated_outputs/test'):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    for file in os.listdir(input_dir):
        if file.endswith('.txt'):
            process_file(os.path.join(input_dir, file), output_dir)

if __name__ == '__main__':
    process_directory()
