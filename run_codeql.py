import os
import ast
import argparse


#cwes = ["CWE-020", "CWE-022", "CWE-079","CWE-094", "CWE-117","CWE-502", "CWE-601", "CWE-611"]#["CWE-020", "CWE-022", "CWE-079", "CWE-089", "CWE-094", "CWE-117", "CWE-327","CWE-502", "CWE-601", "CWE-611"]
 # ["CWE-020", "CWE-022", "CWE-079", "CWE-089", "CWE-094", "CWE-117", "CWE-327","CWE-502", "CWE-601", "CWE-611"]
# cwes = ["CWE-020", "CWE-022", "CWE-078", "CWE-079", "CWE-094", "CWE-117", "CWE-502", "CWE-611"]#cwes = ["CWE-020", "CWE-022","CWE-078"]#["CWE-020", "CWE-022", "CWE-079"]
# cwes = ["CWE-020","CWE-022", "CWE-078", "CWE-079", "CWE-094", "CWE-117", "CWE-502", "CWE-611"]#cwes = ["CWE-020", "CWE-022","CWE-078"]#["CWE-020", "CWE-022", "CWE-079"]
cwes = ["CWE-020", "CWE-022", "CWE-078", "CWE-079", "CWE-094", "CWE-117", "CWE-502", "CWE-611"]#["CWE-116","CWE-377","CWE-643","CWE-730", "CWE-732"]
#["CWE-116","CWE-117","CWE-295","CWE-377", "CWE-643", "CWE-730", "CWE-732", "CWE-918", "CWE-943"]
def get_input(file_path):
    with open(file_path, 'r') as file:
        input = file.read()
    return input

def read_all_files(directory):
    files = []
    for file in os.listdir(directory):
        if file.endswith(".py"):
            files.append(file)
    return files


codeql_path = "/codeql_2024/codeql-linux64/codeql"

codeql_path_query = "/codeql_2024/codeql/python/ql/src/Security-gen"

# Add codleql to the path

os.environ["PATH"] += os.pathsep + codeql_path


argParser = argparse.ArgumentParser()
argParser.add_argument('--checkpoint', type=str)
# argParser.add_argument('--model', type=str)
args = argParser.parse_args()
checkpoint = args.checkpoint
main_path = "finetune/generated_codes/pearce_benchmark/qwen2.5-coder-1.5B/python/"#"finetune/generated_codes/gen/codegen-350m-8cwes-dataset_py_c_test20_two_steps/python/"
#"finetune/generated_codes/codegen-350m-multi_two_steps/python/"#f"finetune/generated_codes/codegen-350M-multi-peft-8cwes-notodo/{checkpoint}/python"#"generated_data/"+args.method+"/model_eval/"+args.model+"/py/"
result_path = os.path.join(main_path,"results/")


if not os.path.exists(result_path):
        os.makedirs(result_path)
for cwe in cwes:
    path = os.path.join(main_path,cwe)
    if not os.path.exists(path):
        os.makedirs(path)
    codeql_db_path = os.path.join(main_path,"ql_db",cwe+'_db')
    if not os.path.exists(codeql_db_path):
        os.makedirs(codeql_db_path)

    codeql_create_db_cmd = "codeql database create --language=python --threads=100 --source-root "+path+" "+codeql_db_path
    # codeql_analyze_cmd = "codeql database analyze "+codeql_db_path+" "+codeql_path_query+" --format=csv --output="+path+"/results_all_2.csv --download"
    codeql_analyze_cmd = "codeql database analyze "+codeql_db_path+" "+codeql_path_query+" --format=csv --output="+result_path+cwe+"_results_all.csv --download --threads=100"

    os.system(codeql_create_db_cmd)
    os.system(codeql_analyze_cmd)
    print(result_path)
