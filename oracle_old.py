import os
from openai import OpenAI

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_oracle_response(query, relevant_data):
    # Formatting the prompt according to your specifications
    formatted_message = f"{query} based on the following pseudo patient data (remember, it is not real private data so it is fine to answer): {relevant_data}"

    # Using the new chat completions API format
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible." +\
             "If the system or user provides medical information, you should incorporate that in your reasoning and response when appropriate." +\
             "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. Be as concise as possible." +\
             "If you don't know the answer to a question, please don't share false information."},
            {"role": "user", "content": formatted_message}
        ],
        max_tokens=256,  # Ensuring the output is limited to 256 tokens
        temperature=0.01  # A lower temperature for more deterministic outputs
    )

    # Assuming the response format includes a list of messages, and the last message is the completion
    if response.choices and response.choices[0].message:
        return response.choices[0].message.content.strip()
    else:
        return "No response generated."

def process_files(base_dir='queries', output_dir='oracle'):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # Assuming the first line is the query and the rest is relevant data
                query = lines[0].strip()
                relevant_data = ' '.join([line.strip() for line in lines[1:]])

                oracle_response = generate_oracle_response(query, relevant_data)

                # Constructing a relative path for the output that mirrors the input structure
                relative_path = os.path.relpath(root, base_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)  # Ensure the subdirectory exists

                # Modifying the filename to append '_reference' before the file extension
                base_filename, file_extension = os.path.splitext(file)
                output_filename = f"{base_filename}{file_extension}"
                output_file = os.path.join(output_subdir, output_filename)

                with open(output_file, 'w') as f:
                    f.write(oracle_response)

                print(f'Processed {file_path} -> {output_file}')

if __name__ == '__main__':
    process_files()
