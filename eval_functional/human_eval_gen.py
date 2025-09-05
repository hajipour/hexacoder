import os
import sys
import torch
import numpy
import random
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from hexacoder.utils import set_seed, separate_imports
from hexacoder.constant import PROMPTS, MODEL_DIRS
from hexacoder.human_eval.problem_yaml import Problem

def load_model(model_name, finetuned_model_name, is_peft):
    model_name = model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache = False, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.bos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_peft:
        finetuned_model_name = finetuned_model_name
        model.load_adapter(finetuned_model_name)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model, torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)

    parser.add_argument('--model_type', type=str, choices=['lm', 'prefix', 'text'], required=True)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--control', type=str, choices=['sec', 'vul'], default='sec')
    parser.add_argument('--is_two', type=bool, default=False)
    parser.add_argument('--temp', type=float, default=0.4)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_gen_len', type=int, default=300)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--num_samples_per_gen', type=int, default=10)

    parser.add_argument('--eval_type', type=str, default='human_eval')
    parser.add_argument('--output_dir', type=str, default='../experiments')
    parser.add_argument('--data_dir', type=str, default='../data_eval')

    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    assert args.num_samples % args.num_samples_per_gen == 0
    args.output_dir = os.path.join(args.output_dir, args.eval_type)
    args.data_dir = os.path.join(args.data_dir, args.eval_type)
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.output_name)
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    shutil.copytree(args.data_dir, args.output_dir)

    return args

args = get_args()

def trim_code(completion, stop_tokens):
    for stop_token in stop_tokens:
        if stop_token in completion:
            completion = completion[:completion.find(stop_token)]
    return completion

def main():
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print("Directory does not exist: {}".format(output_dir))
        sys.exit(1)

    problems = list(
        filter(
            lambda f: not f.name.endswith(".results.yaml"),
            sorted(output_dir.glob("*.yaml")),
        )
    )

    if args.model_type in ('lm', 'text'):
        model_dir = '2b' if args.model_dir is None else args.model_dir
        # if model_dir in MODEL_DIRS:
        #     model_dir = MODEL_DIRS[model_dir]
    else:
        assert args.model_dir is not None
        model_dir = args.model_dir

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device, args.n_gpu)
    model_name = model_dir
    finetuned_model_name = "../finetune/Qwen2.5-Coder-dataset_py_c_test20_float32/checkpoint-255/"
    is_peft = True
    tokenizer, model, device = load_model(model_name, finetuned_model_name, is_peft)
    model.eval()
    if "DeepSeek" in model_name:
        skip_special_tokens = True
    else:
        skip_special_tokens = False
    print(skip_special_tokens)
    for problem_yaml_path in tqdm(problems):
        with problem_yaml_path.open() as f:
            problem = Problem.load(f)
        prompt = problem.prompt
        if args.model_type == 'text':
            if args.control == 'sec':
                prompt = PROMPTS[0] + prompt
            else:
                prompt = PROMPTS[1] + prompt
        
        if args.is_two:
            imports, code = separate_imports(prompt)
            if imports == '':
                prompt = prompt
            else:
                set_seed(args)
                inputs_imports = tokenizer(imports, return_tensors='pt').to(device)
                with torch.no_grad():
                    samples_imports = model.generate(
                        **inputs_imports,
                        do_sample=True,
                        num_return_sequences=1,
                        temperature=args.temp,
                        max_new_tokens=20,
                        top_p=args.top_p,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True
                    )
                    new_imports = tokenizer.decode(samples_imports["input_ids"][0], skip_special_tokens=skip_special_tokens)
                    new_imports, _ = separate_imports(new_imports)
                    
                    # Check if the end of new_imports is a newline and check if the beginning of code is a newline
                    if new_imports[-1] == '\n' or code[0] == '\n':
                        prompt = new_imports+code
                    else:
                        prompt = new_imports+'\n'+code


        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        # if isinstance(model, XGLMForCausalLM):
        #     del inputs['token_type_ids']
        if "incoder" in model_name:
            del inputs['token_type_ids']
        kwargs = dict()
        if args.model_type == 'prefix':
            if args.control == 'sec':
                kwargs['control_id'] = 0
            else:
                kwargs['control_id'] = 1
        for i in range(args.num_samples // args.num_samples_per_gen):
            set_seed(args)
            with torch.no_grad():
                samples = model.generate(
                    **inputs,
                    do_sample=True,
                    num_return_sequences=args.num_samples_per_gen,
                    temperature=args.temp,
                    max_new_tokens=args.max_gen_len,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    **kwargs
                )
            for sample in samples.tolist():
                completion = sample[inputs['input_ids'].shape[1]:]
                if tokenizer.eos_token_id in completion:
                    completion = completion[:completion.index(tokenizer.eos_token_id)]
                completion = tokenizer.decode(completion, skip_special_tokens=skip_special_tokens)
                completion = trim_code(completion, problem.stop_tokens)
                problem.completions.append(completion)
            args.seed += 1
        with problem_yaml_path.open("w") as f:
            f.write(Problem.dump(problem))

if __name__ == '__main__':
    main()
