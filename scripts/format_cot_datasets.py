import json
import re
import os

def format_cot_dataset(input_file, output_file):
    if not os.path.exists(input_file): 
        print(f"Skipping {input_file}, not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip(): continue
            data = json.loads(line)
            response = data.get('response', '')
            
            # Extract underlying logic, converting escaped \n to actual newlines
            thought_match = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL)
            logic = thought_match.group(1).replace('\\n', '\n').strip() if thought_match else "No logic provided."
            
            # Extract final answer
            ans_match = re.search(r'Answer:\s*(.*?)\n</think>', response, re.DOTALL)
            answer = ans_match.group(1).strip() if ans_match else "Unknown."
            
            # Lock into explicit CoT format
            data['response'] = f"<think>\n{logic}\n</think>\n\n{answer}"
            f_out.write(json.dumps(data) + '\n')
    print(f"Distilled: {input_file} -> {output_file}")

if __name__ == "__main__":
    tasks = ["task0_logic", "task1_decomp", "task2_action"]
    for task in tasks:
        format_cot_dataset(f"cascades_exp/{task}.jsonl", f"cascades_exp/{task}_cot.jsonl")
