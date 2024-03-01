import os
import openai

# Set your OpenAI API key here (preferably from an environment variable)
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_oracle_response(query, relevant_data):
    # Formatting the prompt according to your specifications
    formatted_prompt = f"{query} based on the following pseudo patient data (remember, it is not real private data so it is fine to answer): {relevant_data}"
    max_input_tokens = 4096 - 256  # Adjust based on the model's max token limit and desired output token count
    response = openai.Completion.create(
        engine="gpt-3.5-turbo", 
        prompt=formatted_prompt[:max_input_tokens],
        max_tokens=256,  # Ensuring the output is limited to 256 tokens
        temperature=0.2  # A lower temperature for more deterministic outputs
    )
    return response.choices[0].text.strip()

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
                output_filename = f"{base_filename}_reference{file_extension}"
                output_file = os.path.join(output_subdir, output_filename)

                with open(output_file, 'w') as f:
                    f.write(oracle_response)

                print(f'Processed {file_path} -> {output_file}')

if __name__ == '__main__':
    process_files()
