import os
import argparse
import re
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, load_from_disk, disable_caching
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
LANG_FORMATS = {"python":".py", "c":".c"}

# cwes = ["CWE-020", "CWE-022", "CWE-078", "CWE-079", "CWE-117", "CWE-094","CWE-502", "CWE-611"] #"CWE-117"
cwes = ["CWE-116","CWE-117","CWE-295","CWE-377", "CWE-643", "CWE-730", "CWE-732", "CWE-918", "CWE-943"]
# cwes = ["CWE-020", "CWE-022", "CWE-078", "CWE-079", "CWE-094", "CWE-117", "CWE-502", "CWE-611"]
# cwes = ["CWE-117", "CWE-502", "CWE-611"]

# cwes = ["CWE-020", "CWE-022", "CWE-078", "CWE-079", "CWE-094"]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)



def get_input(file_path):
    with open(file_path, 'r') as file:
        input = file.read()
    return input

def set_output(file_path, output):
    with open(file_path, 'w') as file:
        file.write(output)


def separate_imports(code, lang):
    if (lang == "python"):
        # Regular expression to match import statements
        import_pattern = r'^(from\s+\S+\s+import\s+\S+|import\s+\S+).*$'
        
        imports = []
        other_code = []
        
        # Iterate through each line in the code
        for line in code.split('\n'):
            # Strip whitespace from the beginning and end of the line
            stripped_line = line.strip()
            
            # Check if the line matches the import pattern
            if re.match(import_pattern, stripped_line):
                imports.append(line)
            else:
                other_code.append(line)
        
        # Join the imports and other code back into strings
        imports_str = '\n'.join(imports)
        other_code_str = '\n'.join(other_code)
    elif lang == "c":
        # Regular expression to match include statements
        include_pattern = r'^\s*#\s*include\s*(<[^>]+>|"[^"]+")'
        
        includes = []
        other_code = []
        
        # Iterate through each line in the code
        for line in code.split('\n'):
            # Check if the line matches the include pattern
            if re.match(include_pattern, line):
                includes.append(line)
            else:
                other_code.append(line)
        
        # Join the includes and other code back into strings
        imports_str = '\n'.join(includes)
        other_code_str = '\n'.join(other_code)
    else:
        raise ValueError("Unsupported language. Please use 'python' or 'c'.")
    
    return imports_str, other_code_str

def truncate(completion, lang):
    if lang == 'python':
        for match in re.finditer('\n', completion):
            cur_idx, next_idx = match.start(), match.end()
            if next_idx < len(completion) and not completion[next_idx].isspace():
                completion = completion[:cur_idx]
                break
        else:
            last_comment_str = '\n    #'
            if last_comment_str in completion:
                completion = completion[:completion.rfind(last_comment_str)]
    elif lang == 'c':
        if '\n}' in completion:
            completion = completion[:completion.find('\n}')+2]
        else:
            last_comment_strs = ['\n    //', '\n    /*']
            for last_comment_str in last_comment_strs:
                if last_comment_str in completion:
                    completion = completion[:completion.rfind(last_comment_str)]
                    completion = completion.rstrip() + '\n}'

        lines = completion.split('\n')
        final_lines = []
        for line in lines:
            if '->name = "' in line: continue
            final_lines.append(line)
        completion = '\n'.join(final_lines)
    else:
        raise NotImplementedError()

    return completion

def generate_text(model, tokenizer, input, file_path, temperature = 0.4, num_return_sequences = 3, lang = "python"):
    output = model.generate(**input, do_sample=True, max_new_tokens=200, temperature=temperature, 
                    num_return_sequences=num_return_sequences, pad_token_id=tokenizer.pad_token_id, top_p = 0.95, use_cache=True)
                    #no_repeat_ngram_size=6, early_stopping=True

    #end_sequence="###"
    for i in range(num_return_sequences):
        # decoded_input = tokenizer.decode(input["input_ids"][0])
        # decoded_output = tokenizer.decode(output[i])
        # completion_truncated = truncate(decoded_output[len(decoded_input):], lang)
        # decoded_output_tuncated = decoded_input + completion_truncated
        
        decoded_input = tokenizer.decode(input["input_ids"][0], skip_special_tokens=True) # skip_special_tokens=True only for deepseek
        decoded_output = tokenizer.decode(output[i], skip_special_tokens=True) # skip_special_tokens=True only for deepseek
        completion_truncated = truncate(decoded_output[len(decoded_input):], lang)
        decoded_output_tuncated = decoded_output[:len(decoded_input)] + completion_truncated
        set_output(file_path +"_" +str(i) + prog_format, decoded_output_tuncated)

def generate_text_temp_qwen(model, tokenizer, input, file_path, temperature = 0.4, num_return_sequences = 3, lang = "python"):
    # Map input from str to mapping
    # input = tokenizer(input, return_tensors="pt").to(model.device)
    output = model.generate(**input, do_sample=True, max_new_tokens=200, temperature=temperature, 
                    num_return_sequences=num_return_sequences, pad_token_id=tokenizer.pad_token_id, top_p = 0.95, use_cache=True)
                    #no_repeat_ngram_size=6, early_stopping=True

    #end_sequence="###"
    for i in range(num_return_sequences):
        # decoded_input = tokenizer.decode(input["input_ids"][0])
        # decoded_output = tokenizer.decode(output[i])
        # completion_truncated = truncate(decoded_output[len(decoded_input):], lang)
        # decoded_output_tuncated = decoded_input + completion_truncated
        
        decoded_input = tokenizer.decode(input["input_ids"][0], skip_special_tokens=True) # skip_special_tokens=True only for deepseek
        decoded_output = tokenizer.decode(output[i], skip_special_tokens=True) # skip_special_tokens=True only for deepseek
        completion_truncated = truncate(decoded_output[len(decoded_input):], lang)
        decoded_output_tuncated = decoded_output[:len(decoded_input)] + completion_truncated
        set_output(file_path +"_" +str(i) + prog_format, decoded_output_tuncated)

def generate_text2(model, tokenizer, input, imports, file_path, temperature = 0.4, num_return_sequences = 3, lang = "python"):
    imports = model.generate(**imports, do_sample=True, max_new_tokens=20, temperature=temperature, 
                    num_return_sequences=1, pad_token_id=tokenizer.pad_token_id, top_p = 0.95, use_cache=True)

    decoded_imports = tokenizer.decode(imports[0], skip_special_tokens=True) # skip_special_tokens=True only for deepseek
    imports_str, _ = separate_imports(decoded_imports, lang)
    decoded_input = tokenizer.decode(input["input_ids"][0], skip_special_tokens=True) # skip_special_tokens=True only for deepseek
    input_str = imports_str +"\n"+ decoded_input
    input_new = tokenizer(input_str, return_tensors="pt").to(model.device)
    output = model.generate(**input_new, do_sample=True, max_new_tokens=180, temperature=temperature, 
                    num_return_sequences=num_return_sequences, pad_token_id=tokenizer.pad_token_id, top_p = 0.95, use_cache=True)
                    #no_repeat_ngram_size=6, early_stopping=True

    #end_sequence="###"
    for i in range(num_return_sequences):
        decoded_output = tokenizer.decode(output[i], skip_special_tokens=True) # skip_special_tokens=True only for deepseek
        completion_truncated = truncate(decoded_output[len(input_str):], lang)
        decoded_output_tuncated = decoded_output[:len(input_str)] + completion_truncated
        set_output(file_path +"_" +str(i) + prog_format, decoded_output_tuncated)

def generate_text2_temp_qwen(model, tokenizer, input, imports, file_path, temperature = 0.4, num_return_sequences = 3, lang = "python"):
    # input = tokenizer(input, return_tensors="pt").to(model.device)
    imports = model.generate(**imports, do_sample=True, max_new_tokens=20, temperature=temperature, 
                    num_return_sequences=1, pad_token_id=tokenizer.pad_token_id, top_p = 0.95, use_cache=True)

    decoded_imports = tokenizer.decode(imports[0], skip_special_tokens=True) # skip_special_tokens=True only for deepseek
    imports_str, _ = separate_imports(decoded_imports, lang)
    decoded_input = tokenizer.decode(input["input_ids"][0], skip_special_tokens=True) # skip_special_tokens=True only for deepseek
    input_str = imports_str +"\n"+ decoded_input
    input_new = tokenizer(input_str, return_tensors="pt").to(model.device)
    output = model.generate(**input_new, do_sample=True, max_new_tokens=180, temperature=temperature, 
                    num_return_sequences=num_return_sequences, pad_token_id=tokenizer.pad_token_id, top_p = 0.95, use_cache=True)
                    #no_repeat_ngram_size=6, early_stopping=True

    #end_sequence="###"
    for i in range(num_return_sequences):
        decoded_output = tokenizer.decode(output[i], skip_special_tokens=True) # skip_special_tokens=True only for deepseek
        completion_truncated = truncate(decoded_output[len(input_str):], lang)
        decoded_output_tuncated = decoded_output[:len(input_str)] + completion_truncated
        set_output(file_path +"_" +str(i) + prog_format, decoded_output_tuncated)

def generate_text2_incoder(model, tokenizer, input, infill, file_path, temperature = 0.4, num_return_sequences = 3, lang = "python"):
    # signals the end of a generated infill
    EOM = "<|endofmask|>"
    
    imports = model.generate(**infill, do_sample=True, max_new_tokens=20, temperature=temperature, 
                    num_return_sequences=1, pad_token_id=tokenizer.pad_token_id, top_p = 0.95, use_cache=True)

    decoded_imports = tokenizer.decode(imports[0], clean_up_tokenization_spaces=False, skip_special_tokens=True)
      
    imports_str, _ = separate_imports(decoded_imports, lang)

    decoded_input = tokenizer.decode(input["input_ids"][0], clean_up_tokenization_spaces=False, skip_special_tokens=True)
    input_str = imports_str +"\n"+ decoded_input
    input_new = tokenizer(input_str, return_token_type_ids=False, return_tensors="pt").to(model.device)
    output = model.generate(**input_new, do_sample=True, max_new_tokens=180, temperature=temperature, 
                    num_return_sequences=num_return_sequences, pad_token_id=tokenizer.pad_token_id, top_p = 0.95, use_cache=True)
                    #no_repeat_ngram_size=6, early_stopping=True

    for i in range(num_return_sequences):
        decoded_output = tokenizer.decode(output[i], clean_up_tokenization_spaces=False, skip_special_tokens=True)
        completion_truncated = truncate(decoded_output[len(input_str):], lang)
        decoded_output_tuncated = decoded_output[:len(input_str)] + completion_truncated
        set_output(file_path +"_" +str(i) + prog_format, decoded_output_tuncated)

def generate_text_incoder(model, tokenizer, input, file_path, temperature = 0.4, num_return_sequences = 3, lang = "python"):
    output = model.generate(**input, do_sample=True, max_new_tokens=200, temperature=temperature, 
                    num_return_sequences=num_return_sequences, pad_token_id=tokenizer.pad_token_id, top_p = 0.95, use_cache=True)
                    #no_repeat_ngram_size=6, early_stopping=True

    #end_sequence="###"
    for i in range(num_return_sequences):
        decoded_input = tokenizer.decode(input["input_ids"][0], clean_up_tokenization_spaces=False, skip_special_tokens=True)

        decoded_output = tokenizer.decode(output[i], clean_up_tokenization_spaces=False, skip_special_tokens=True)
        completion_truncated = truncate(decoded_output[len(decoded_input):], lang)
        decoded_output_tuncated = decoded_output[:len(decoded_input)] + completion_truncated
        set_output(file_path +"_" +str(i) + prog_format, decoded_output_tuncated)

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--lang', type=str, default="python")
    argParser.add_argument('--temperature', type=float, default=0.4)
    argParser.add_argument('--benchmark', type=str, choices=["codelmsec", "pearce", "gen"], default="codelmsec")
    argParser.add_argument('--is_baseline', type=bool, default=False)
    argParser.add_argument('--seed', type=int, default=42)
    args = argParser.parse_args()

    lang = args.lang
    is_baseline = args.is_baseline
    set_seed(args)

    dataset = load_from_disk("dataset") 
    model_dir = "../models/"
    model_name = "Qwen2.5-Coder-1.5B/"#"../models/codegen-2B-multi/"
    model_path = os.path.join(model_dir, model_name)
    finetuned_model_path = "Qwen2.5-Coder-dataset_py_c_test20_float32/checkpoint-255/"
    if model_name == "incoder-6B":
        print("Loading incoder-6B model")
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, device_map='auto')
        # revision="float16", torch_dtype=torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_name == "DeepSeek-Coder-V2-Lite-Base":
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache = False, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, use_cache = False, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Load the LoRA adapter
    if not is_baseline:
        model.load_adapter(finetuned_model_path)
    print(model)
    prog_format = LANG_FORMATS[lang]
    output_dir = "generated_codes/pearce_benchmark/qwen2.5-coder-1.5B/"
    benchmark = args.benchmark
    if benchmark == "codelmsec":
        num_return_sequences = 5
        prompt_path = os.path.join("../benchmark_data/codelmsec/",lang)
        output_path = os.path.join(output_dir,lang)
        if lang == "c":
            cwes = ["CWE-022","CWE-190","CWE-476", "CWE-787"]
    elif benchmark == "pearce":
        num_return_sequences = 15
        prompt_path = os.path.join("../benchmark_data/pearce/",lang)
        if lang == "python":
            cwes = ["CWE-020", "CWE-022", "CWE-078", "CWE-079", "CWE-094", "CWE-502", "CWE-611"]
        elif lang == "c":
            cwes = ["CWE-022", "CWE-190","CWE-476", "CWE-787"]

        output_path = os.path.join(output_dir,lang)
    elif benchmark == "gen":
        num_return_sequences = 15
        prompt_path = os.path.join("../benchmark_data/gen/",lang)
        output_path = os.path.join(output_dir,lang)
        if lang == "python":
            cwes = ["CWE-116","CWE-117","CWE-295","CWE-377", "CWE-643", "CWE-730", "CWE-732", "CWE-918", "CWE-943"]
        # elif lang == "c":
        #     cwes = ["CWE-022", "CWE-190","CWE-476", "CWE-787"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cnt_cwe = 0
    for cwe in cwes:
        cur_output_path = os.path.join(output_path,cwe)
        cur_prompt_path  = os.path.join(prompt_path,cwe)
        # read all file names  of cur_prompt_path
        cur_prompt_file_names = []
        for file in os.listdir(cur_prompt_path):
            if file.endswith(prog_format):
                cur_prompt_file_names.append(file)

        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)
        for file in cur_prompt_file_names:

                # print(cur_prompt_path+file)
                print(file)
                input = get_input(cur_prompt_path+"/"+file)
                file_name_without_py = file.split(".")[0]
                # Seperate imports and code in the input
                imports, code = separate_imports(input, lang)
                if "incoder" in model_name:
                    infill = imports#+"\n"+"<|mask:0|>"+"\n"+code+"<|mask:0|>"
                    infill = tokenizer(infill, return_token_type_ids=False, return_tensors="pt").to(model.device)
                    code = tokenizer(code, return_token_type_ids=False, return_tensors="pt").to(model.device)
                    # input = tokenizer(input, return_token_type_ids=False, return_tensors="pt").to(model.device)
                    
                    generate_text2_incoder(model, tokenizer, code, infill, cur_output_path +'/'+file_name_without_py, temperature = args.temperature, num_return_sequences = num_return_sequences, lang = lang)
                    # generate_text_incoder(model, tokenizer, input, cur_output_path +'/'+file_name_without_py, temperature = args.temperature, num_return_sequences = num_return_sequences, lang = lang)
                    
                else:
                    # imports = tokenizer(imports, return_tensors="pt").to(model.device)
                    # code = tokenizer(code, return_tensors="pt").to(model.device)
                    input = tokenizer(input, return_tensors="pt").to(model.device)
                    generate_text_temp_qwen(model, tokenizer, input, cur_output_path +'/'+file_name_without_py, temperature = args.temperature, num_return_sequences = num_return_sequences, lang = lang)
                    # generate_text2_temp_qwen(model, tokenizer, code, imports, cur_output_path +'/'+file_name_without_py, temperature = args.temperature, num_return_sequences = num_return_sequences, lang = lang)

                    # generate_text(model, tokenizer, input, cur_output_path +'/'+file_name_without_py, temperature = args.temperature, num_return_sequences = num_return_sequences, lang = lang)
        cnt_cwe += 1
