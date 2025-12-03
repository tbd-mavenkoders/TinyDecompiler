import yaml
from pathlib import Path
from ..utils.reassembler import Reassembler
from ..utils.llm_interface import create_llm_interface, clean_llm_output
from ..utils.compile import Compiler,OptimizationLevel
from ..utils.clean_errors import ErrorNormalizer
import re
import shutil
import tempfile
import os
from typing import Tuple, List, Dict
import json


c = Compiler()


# Config.yaml paths
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
print(f"Loading config from: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    
llm_interface = create_llm_interface(
  provider=config["llm"]["gemini_provider"],
  model_name=config["llm"]["gemini_model_name"],
  api_key=config["llm"]["gemini_api_key"]
)
corpus_root = Path(config["paths"]["humaneval_corpus_root"])
output_dir = Path(config["paths"]["humaneval_static_repair_path"])
output_dir.mkdir(parents=True, exist_ok=True)

def get_initial_prompt(c_code: str, summary: str,language: str) -> str:
  """
  Generate the initial prompt for the repair tool given C code of the particular function.
  """
  initial_prompt = config["prompts"]["system_prompt"]
  prompt = f"{initial_prompt}\n\n```Language:{language}\nSummary:{summary}\n{c_code}\n```"
  return prompt

def summarize_function(ghidra_code: str, asm: str, language: str) -> str:
  """
  Summarize the function purpose using LLM.
  """
  summary_prompt = config["prompts"]["summary_prompt"]
  prompt = f"{summary_prompt}\n\n```Language:{language}\nCode:{ghidra_code}\nASM:{asm}\n```"
  summary = llm_interface.generate(prompt)
  return clean_llm_output(summary)



def get_optimized_code(original_c_code: str, summary: str, language: str, max_iterations: int) -> str:
  """
  Generate optimized C code using LLM for the given original C code file.
  """
  compilable_code = original_c_code
  
  # handle everything in a temporary directory
  with tempfile.TemporaryDirectory() as temp_dir:
    # write the original code to a file
    original_c_file = Path(temp_dir) / "original.c"
    with open(original_c_file, "w") as f:
      f.write(original_c_code)
    
    # check if original code compiles
    status, message = c.compile_source(
      source_file_path = original_c_file,
      output_file_path = Path(temp_dir) / "original.out",
      opt = OptimizationLevel.O0,
      is_cpp = (language == "cpp")
    )
  
    if status:
      print("Original Ghidra Code compiles successfully. No optimization needed.")
      return True, original_c_code

    
    # if not, provide an initial LLM optimization
    print("Original Ghidra Code does not compile. Starting optimization...")
    initial_prompt = get_initial_prompt(original_c_code,summary,language)
    optimized_code = llm_interface.generate(initial_prompt)
    
    # check if initially prompted code compiles
    original_c_file.write_text(optimized_code)
    status, message = c.compile_source(
      source_file_path = original_c_file,
      output_file_path = Path(temp_dir) / "optimized.out",
      opt = OptimizationLevel.O0,
      is_cpp = (language == "cpp")
    )
    
    if status:
      print("Optimized code compiles successfully after initial LLM prompt.")
      return True, optimized_code
    
    # begin static repair loop
    for iteration in range(max_iterations):
      print(f"Static Repair Iteration {iteration + 1}...")
      
      # acquire optimized output through error passing
      e = ErrorNormalizer()
      error_prompt = e.format_for_llm(message)
      repair_prompt = f"{config['prompts']['compilation_error']}\n\n```c\nLanguage:{language}\nSummary:{summary}\nCode:{optimized_code}\n```\n\nCompilation Errors:\n{error_prompt}\n\nPlease provide the corrected C code."
      optimized_code = llm_interface.generate(repair_prompt)
      
      # check if it compiles
      original_c_file.write_text(optimized_code)
      status, message = c.compile_source(
        source_file_path = original_c_file,
        output_file_path = Path(temp_dir) / "optimized.out",
        opt = OptimizationLevel.O0,
        is_cpp = (language == "cpp")
      )
      
      if status:
        print(f"Optimized code compiles successfully after {iteration + 1} iterations.")
        return True, optimized_code
      else:
        print(f"Optimized code still does not compile after iteration {iteration + 1}. Continuing...")
    
      
    print("Max optimization iterations reached. Returning last optimized code.")
    return False, optimized_code
  
def process_json_file(json_file_path: Path, max_iterations: int) -> Dict:
  success = 0
  total = 0
  """
  Process a single JSON file containing decompiled C code and attempt to optimize it.
      {
        "index": 0,
        "func_name": "func0",
        "func_dep": "#include<stdio.h>\n#include<vector>\n#include<math.h>\nusing namespace std;\n#include<algorithm>\n#include<stdlib.h>\n\n",
        "func": "bool func0(vector<float> numbers, float threshold){\n    int i,j;\n    \n    for (i=0;i<numbers.size();i++)\n    for (j=i+1;j<numbers.size();j++)\n    if (abs(numbers[i]-numbers[j])<threshold)\n    return true;\n\n    return false;\n}\n\n",
        "test": "#undef NDEBUG\n\n#include<assert.h>\nint main(){\n    vector<float> a={1.0, 2.0, 3.9, 4.0, 5.0, 2.2};\n    assert (func0(a, 0.3)==true);\n    assert (func0(a, 0.05) == false);\n\n    assert (func0({1.0, 2.0, 5.9, 4.0, 5.0}, 0.95) == true);\n    assert (func0({1.0, 2.0, 5.9, 4.0, 5.0}, 0.8) ==false);\n    assert (func0({1.0, 2.0, 3.0, 4.0, 5.0}, 2.0) == true);\n    assert (func0({1.1, 2.2, 3.1, 4.1, 5.1}, 1.0) == true);\n    assert (func0({1.1, 2.2, 3.1, 4.1, 5.1}, 0.5) == false);\n    assert (func0({1.0, 2.0, 3.0}, 0.5) == false && \"failure 1\");\n    assert (func0({1.0, 2.8, 3.0, 4.0, 5.0, 2.0}, 0.3) && \"failure 2\") ;\n}\n",
        "opt": "O0",
        "language": "cpp",
        "asm": "func0(std::vector<float, std::allocator<float> >, float):\nendbr64\npush   %rbp\nmov    %rsp,%rbp\npush   %rbx\nsub    $0x28,%rsp\nmov    %rdi,-0x28(%rbp)\nmovss  %xmm0,-0x2c(%rbp)\nmovl   $0x0,-0x18(%rbp)\nmov    -0x18(%rbp),%eax\nmovslq %eax,%rbx\nmov    -0x28(%rbp),%rax\nmov    %rax,%rdi\ncallq  1cbc <_ZNKSt6vectorIfSaIfEE4sizeEv>\ncmp    %rax,%rbx\nsetb   %al\ntest   %al,%al\nje     12f8 <_Z5func0St6vectorIfSaIfEEf+0xcf>\nmov    -0x18(%rbp),%eax\nadd    $0x1,%eax\nmov    %eax,-0x14(%rbp)\nmov    -0x14(%rbp),%eax\nmovslq %eax,%rbx\nmov    -0x28(%rbp),%rax\nmov    %rax,%rdi\ncallq  1cbc <_ZNKSt6vectorIfSaIfEE4sizeEv>\ncmp    %rax,%rbx\nsetb   %al\ntest   %al,%al\nje     12ef <_Z5func0St6vectorIfSaIfEEf+0xc6>\nmov    -0x18(%rbp),%eax\nmovslq %eax,%rdx\nmov    -0x28(%rbp),%rax\nmov    %rdx,%rsi\nmov    %rax,%rdi\ncallq  1ce4 <_ZNSt6vectorIfSaIfEEixEm>\nmovss  (%rax),%xmm2\nmovss  %xmm2,-0x30(%rbp)\nmov    -0x14(%rbp),%eax\nmovslq %eax,%rdx\nmov    -0x28(%rbp),%rax\nmov    %rdx,%rsi\nmov    %rax,%rdi\ncallq  1ce4 <_ZNSt6vectorIfSaIfEEixEm>\nmovss  (%rax),%xmm0\nmovss  -0x30(%rbp),%xmm2\nsubss  %xmm0,%xmm2\nmovaps %xmm2,%xmm0\ncallq  1c6d <_ZSt3absf>\nmovss  -0x2c(%rbp),%xmm1\ncomiss %xmm0,%xmm1\nseta   %al\ntest   %al,%al\nje     12e9 <_Z5func0St6vectorIfSaIfEEf+0xc0>\nmov    $0x1,%eax\njmp    12fd <_Z5func0St6vectorIfSaIfEEf+0xd4>\naddl   $0x1,-0x14(%rbp)\njmp    126f <_Z5func0St6vectorIfSaIfEEf+0x46>\naddl   $0x1,-0x18(%rbp)\njmpq   1246 <_Z5func0St6vectorIfSaIfEEf+0x1d>\nmov    $0x0,%eax\nadd    $0x28,%rsp\npop    %rbx\npop    %rbp\nretq\n",
        "ida_asm": "_Z5func0St6vectorIfSaIfEEf:\nendbr64\npush    rbp\nmov     rbp, rsp\npush    rbx\nsub     rsp, 28h\nmov     [rbp+var_28], rdi\nmovss   [rbp+var_2C], xmm0\nmov     [rbp+var_18], 0\njmp     loc_1301\nloc_126B:\nmov     eax, [rbp+var_18]\nadd     eax, 1\nmov     [rbp+var_14], eax\njmp     short loc_12DD\nloc_1276:\nmov     eax, [rbp+var_18]\nmovsxd  rdx, eax\nmov     rax, [rbp+var_28]\nmov     rsi, rdx\nmov     rdi, rax\ncall    _ZNSt6vectorIfSaIfEEixEm; std::vector<float>::operator[](ulong)\nmovss   xmm2, dword ptr [rax]\nmovss   [rbp+var_30], xmm2\nmov     eax, [rbp+var_14]\nmovsxd  rdx, eax\nmov     rax, [rbp+var_28]\nmov     rsi, rdx\nmov     rdi, rax\ncall    _ZNSt6vectorIfSaIfEEixEm; std::vector<float>::operator[](ulong)\nmovss   xmm0, dword ptr [rax]\nmovss   xmm2, [rbp+var_30]\nsubss   xmm2, xmm0\nmovd    eax, xmm2\nmovd    xmm0, eax; float\ncall    _ZSt3absf; std::abs(float)\nmovss   xmm1, [rbp+var_2C]\ncomiss  xmm1, xmm0\nsetnbe  al\ntest    al, al\njz      short loc_12D9\nmov     eax, 1\njmp     short loc_1326\nloc_12D9:\nadd     [rbp+var_14], 1\nloc_12DD:\nmov     eax, [rbp+var_14]\nmovsxd  rbx, eax\nmov     rax, [rbp+var_28]\nmov     rdi, rax\ncall    _ZNKSt6vectorIfSaIfEE4sizeEv; std::vector<float>::size(void)\ncmp     rbx, rax\nsetb    al\ntest    al, al\njnz     loc_1276\nadd     [rbp+var_18], 1\nloc_1301:\nmov     eax, [rbp+var_18]\nmovsxd  rbx, eax\nmov     rax, [rbp+var_28]\nmov     rdi, rax\ncall    _ZNKSt6vectorIfSaIfEE4sizeEv; std::vector<float>::size(void)\ncmp     rbx, rax\nsetb    al\ntest    al, al\njnz     loc_126B\nmov     eax, 0\nloc_1326:\nmov     rbx, [rbp+var_8]\nleave\nretn",
        "ida_pseudo": "long long  func0(long long a1, float a2)\n{\n  __m128i v2; // xmm2\n  float v3; // xmm0_4\n  float v5; // [rsp+0h] [rbp-30h]\n  int i; // [rsp+18h] [rbp-18h]\n  int j; // [rsp+1Ch] [rbp-14h]\n\n  for ( i = 0; i < (unsigned long long)std::vector<float>::size(a1); ++i )\n  {\n    for ( j = i + 1; j < (unsigned long long)std::vector<float>::size(a1); ++j )\n    {\n      v5 = *(float *)std::vector<float>::operator[](a1, i);\n      v2 = (__m128i)LODWORD(v5);\n      *(float *)v2.m128i_i32 = v5 - *(float *)std::vector<float>::operator[](a1, j);\n      v3 = COERCE_FLOAT(_mm_cvtsi128_si32(v2));\n      std::abs(v3);\n      if ( a2 > v3 )\n        return 1LL;\n    }\n  }\n  return 0LL;\n}",
        "ghidra_asm": "func0:\nENDBR64\nPUSH RBP\nMOV RBP,RSP\nPUSH RBX\nSUB RSP,0x28\nMOV qword ptr [RBP + -0x28],RDI\nMOVSS dword ptr [RBP + -0x2c],XMM0\nMOV dword ptr [RBP + -0x18],0x0\nJMP 0x00101301\nLAB_0010126b:\nMOV EAX,dword ptr [RBP + -0x18]\nADD EAX,0x1\nMOV dword ptr [RBP + -0x14],EAX\nJMP 0x001012dd\nLAB_00101276:\nMOV EAX,dword ptr [RBP + -0x18]\nMOVSXD RDX,EAX\nMOV RAX,qword ptr [RBP + -0x28]\nMOV RSI,RDX\nMOV RDI,RAX\nCALL 0x00101d6a\nMOVSS XMM2,dword ptr [RAX]\nMOVSS dword ptr [RBP + -0x30],XMM2\nMOV EAX,dword ptr [RBP + -0x14]\nMOVSXD RDX,EAX\nMOV RAX,qword ptr [RBP + -0x28]\nMOV RSI,RDX\nMOV RDI,RAX\nCALL 0x00101d6a\nMOVSS XMM0,dword ptr [RAX]\nMOVSS XMM2,dword ptr [RBP + -0x30]\nSUBSS XMM2,XMM0\nMOVD EAX,XMM2\nMOVD XMM0,EAX\nCALL 0x00101d22\nMOVSS XMM1,dword ptr [RBP + -0x2c]\nCOMISS XMM1,XMM0\nSETA AL\nTEST AL,AL\nJZ 0x001012d9\nMOV EAX,0x1\nJMP 0x00101326\nLAB_001012d9:\nADD dword ptr [RBP + -0x14],0x1\nLAB_001012dd:\nMOV EAX,dword ptr [RBP + -0x14]\nMOVSXD RBX,EAX\nMOV RAX,qword ptr [RBP + -0x28]\nMOV RDI,RAX\nCALL 0x00101d42\nCMP RBX,RAX\nSETC AL\nTEST AL,AL\nJNZ 0x00101276\nADD dword ptr [RBP + -0x18],0x1\nLAB_00101301:\nMOV EAX,dword ptr [RBP + -0x18]\nMOVSXD RBX,EAX\nMOV RAX,qword ptr [RBP + -0x28]\nMOV RDI,RAX\nCALL 0x00101d42\nCMP RBX,RAX\nSETC AL\nTEST AL,AL\nJNZ 0x0010126b\nMOV EAX,0x0\nLAB_00101326:\nMOV RBX,qword ptr [RBP + -0x8]\nLEAVE\nRET",
        "ghidra_pseudo": "/* func0(std::vector<float, std::allocator<float> >, float) */\n\nint8 func0(vector param_1,float param_2)\n\n{\n  float *pfVar1;\n  ulong uVar2;\n  int4 in_register_0000003c;\n  vector<float,std::allocator<float>> *this;\n  float fVar3;\n  int local_20;\n  int local_1c;\n  \n  this = (vector<float,std::allocator<float>> *)CONCAT44(in_register_0000003c,param_1);\n  local_20 = 0;\n  do {\n    uVar2 = std::vector<float,std::allocator<float>>::size(this);\n    local_1c = local_20;\n    if (uVar2 <= (ulong)(long)local_20) {\n      return 0;\n    }\n    while( true ) {\n      local_1c = local_1c + 1;\n      uVar2 = std::vector<float,std::allocator<float>>::size(this);\n      if (uVar2 <= (ulong)(long)local_1c) break;\n      pfVar1 = (float *)std::vector<float,std::allocator<float>>::operator[](this,(long)local_20);\n      fVar3 = *pfVar1;\n      pfVar1 = (float *)std::vector<float,std::allocator<float>>::operator[](this,(long)local_1c);\n      fVar3 = (float)std::abs(fVar3 - *pfVar1);\n      if (fVar3 < param_2) {\n        return 1;\n      }\n    }\n    local_20 = local_20 + 1;\n  } while( true );\n}"
    },
  """
  # Load the JSON file
  with open(json_file_path, "r") as f:
    dataset = json.load(f)
    
  # this json file contains the entire dataset, so iterate through it
  for data in dataset:
    total += 1
    original_c_code = data["ghidra_pseudo"]
    language = data["language"]
    asm = data["asm"]
    summary = summarize_function(original_c_code,asm,language)
    status, optimized_code = get_optimized_code(original_c_code,summary,language,max_iterations)
    if status:
      success += 1
      print(f"Function {data['func_name']} optimized successfully.")
    else:
      print(f"Function {data['func_name']} optimization failed after max iterations.")
      
    data["summary"] = summary
    data["optimized_func"] = optimized_code
    data["optimization_status"] = status
      
    output_file_path = output_dir / json_file_path.name
    
    # append to output JSON file
    if output_file_path.exists():
      with open(output_file_path, "r") as f:
        existing_data = json.load(f)
      existing_data.append(data)
      with open(output_file_path, "w") as f:
        json.dump(existing_data, f, indent=4)
    else:
      with open(output_file_path, "w") as f:
        json.dump([data], f, indent=4)
    
  return success, total
    
    
def main():
  """
  Main function to process all JSON files in the corpus root directory.
  """
  max_iterations = config["static_repair"]["max_iterations"]
  
  json_files = list(corpus_root.glob("*.json"))
  print(f"Found {len(json_files)} JSON files to process.")
  
  for json_file in json_files:
    print(f"Processing file: {json_file.name}")
    success, total = process_json_file(json_file, max_iterations)
    
  # summary of compilation rates
  print(f"Total Functions Processed: {total}")
  print(f"Total Functions Successfully Optimized: {success}")
  print(f"Overall Optimization Success Rate: {success / total:.4f}")
  

if __name__ == "__main__":
    main()

  

  
  

  
    
  
    
  
  
  
  
      
      
    
  
    
    
  
  
  
