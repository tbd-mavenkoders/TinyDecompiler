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
import subprocess


c = Compiler()


# Config.yaml paths
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
print(f"Loading config from: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

static_repaired_root = Path(config["paths"]["humaneval_static_repair_path"])


def compile_and_execute(c_file_path: str, language: str) -> Tuple[bool, bool, str]:
  """
  Compile and execute the C code, returning any runtime errors.
  """
  output_executable = c_file_path.with_suffix('')
  status, compile_message = c.compile_source(
    source_file_path = c_file_path,
    output_file_path = output_executable,
    opt = OptimizationLevel.O0,
    is_cpp = (language == "cpp")
  )
  # if fails to compile, return error
  if not status:
    return False, False, compile_message
  # if compiles, run and capture output by running ./output_executable
  try:
    command = [f"./{output_executable.name}"]
    print("Executing command:", " ".join(command))
    res = subprocess.run(command, cwd = output_executable.parent, capture_output=True, text=True, timeout=5)
    if res.returncode == 0:
      return True, True, res.stdout
    else:
      return True, False, res.stderr
  except Exception as e:
    return True, False, str(e)
  


  
def process_json_file(json_file_path: Path) -> Dict:
  ce_success = 0
  c_fail = 0
  e_fail = 0
  total = 0
  """
  Process a single JSON file containing decompiled C code and attempt to optimize it.
    {
        "index": 730,
        "func_name": "func0",
        "func_dep": "#include <stdio.h>\n#include <string.h>\n",
        "func": "int func0(const char *str, const char *substring) {\n    int out = 0;\n    int str_len = strlen(str);\n    int sub_len = strlen(substring);\n    if (str_len == 0) return 0;\n    for (int i = 0; i <= str_len - sub_len; i++) {\n        if (strncmp(&str[i], substring, sub_len) == 0)\n            out++;\n    }\n    return out;\n}",
        "test": "#include <assert.h>\n\nint main() {\n    assert(func0(\"\", \"x\") == 0);\n    assert(func0(\"xyxyxyx\", \"x\") == 4);\n    assert(func0(\"cacacacac\", \"cac\") == 4);\n    assert(func0(\"john doe\", \"john\") == 1);\n\n    return 0;\n}",
        "opt": "O2",
        "language": "c",
        "asm": "func0:\nendbr64\npush   %r14\npush   %r13\npush   %r12\nmov    %rsi,%r12\npush   %rbp\npush   %rbx\nmov    %rdi,%rbx\ncallq  1080 <strlen@plt>\nmov    %eax,%r14d\ntest   %eax,%eax\nje     12de <func0+0x5e>\nmov    %r12,%rdi\nmov    %rax,%r13\ncallq  1080 <strlen@plt>\nsub    %eax,%r13d\njs     12f0 <func0+0x70>\nmovslq %r13d,%r13\nmovslq %eax,%rbp\nxor    %r14d,%r14d\nlea    0x1(%rbx,%r13,1),%r13\nnopl   0x0(%rax)\nmov    %rbx,%rdi\nmov    %rbp,%rdx\nmov    %r12,%rsi\ncallq  1070 <strncmp@plt>\ncmp    $0x1,%eax\nadc    $0x0,%r14d\nadd    $0x1,%rbx\ncmp    %r13,%rbx\njne    12c0 <func0+0x40>\npop    %rbx\nmov    %r14d,%eax\npop    %rbp\npop    %r12\npop    %r13\npop    %r14\nretq\nnopw   0x0(%rax,%rax,1)\nxor    %r14d,%r14d\njmp    12de <func0+0x5e>\nnopw   %cs:0x0(%rax,%rax,1)\n",
        "ida_asm": "func0:\nendbr64\npush    r14\npush    r13\nmov     r13, rsi\npush    r12\npush    rbp\npush    rbx\nmov     rbx, rdi\ncall    _strlen\nmov     ebp, eax\ntest    eax, eax\njz      short loc_12DD\nmov     rdi, r13; s\nmov     r14, rax\ncall    _strlen\nsub     r14d, eax\njs      short loc_12F0\nmovsxd  r14, r14d\nmovsxd  r12, eax\nxor     ebp, ebp\nlea     r14, [rbx+r14+1]\nnop     word ptr [rax+rax+00h]\nloc_12C0:\nmov     rdi, rbx; s1\nmov     rdx, r12; n\nmov     rsi, r13; s2\ncall    _strncmp\ncmp     eax, 1\nadc     ebp, 0\nadd     rbx, 1\ncmp     rbx, r14\njnz     short loc_12C0\nloc_12DD:\npop     rbx\nmov     eax, ebp\npop     rbp\npop     r12\npop     r13\npop     r14\nretn\nloc_12F0:\nxor     ebp, ebp\njmp     short loc_12DD",
        "ida_pseudo": "long long  func0(char *s1, char *s2)\n{\n  const char *v2; // rbx\n  unsigned int v3; // eax\n  unsigned int v4; // ebp\n  unsigned int v5; // r14d\n  int v6; // eax\n  int v7; // r14d\n  size_t v8; // r12\n  char *v9; // r14\n\n  v2 = s1;\n  v3 = strlen(s1);\n  v4 = v3;\n  if ( v3 )\n  {\n    v5 = v3;\n    v6 = strlen(s2);\n    v7 = v5 - v6;\n    if ( v7 < 0 )\n    {\n      return 0;\n    }\n    else\n    {\n      v8 = v6;\n      v4 = 0;\n      v9 = &s1[v7 + 1];\n      do\n        v4 += strncmp(v2++, s2, v8) == 0;\n      while ( v2 != v9 );\n    }\n  }\n  return v4;\n}",
        "ghidra_asm": "func0:\nENDBR64\nPUSH R14\nPUSH R13\nMOV R13,RSI\nPUSH R12\nPUSH RBP\nPUSH RBX\nMOV RBX,RDI\nCALL 0x00101080\nMOV EBP,EAX\nTEST EAX,EAX\nJZ 0x001012dd\nMOV RDI,R13\nMOV R14,RAX\nCALL 0x00101080\nSUB R14D,EAX\nJS 0x001012f0\nMOVSXD R14,R14D\nMOVSXD R12,EAX\nXOR EBP,EBP\nLEA R14,[RBX + R14*0x1 + 0x1]\nNOP word ptr [RAX + RAX*0x1]\nLAB_001012c0:\nMOV RDI,RBX\nMOV RDX,R12\nMOV RSI,R13\nCALL 0x00101070\nCMP EAX,0x1\nADC EBP,0x0\nADD RBX,0x1\nCMP RBX,R14\nJNZ 0x001012c0\nLAB_001012dd:\nPOP RBX\nMOV EAX,EBP\nPOP RBP\nPOP R12\nPOP R13\nPOP R14\nRET\nLAB_001012f0:\nXOR EBP,EBP\nJMP 0x001012dd",
        "ghidra_pseudo": "ulong func0(char *param_1,char *param_2)\n\n{\n  char *pcVar1;\n  size_t sVar2;\n  size_t sVar3;\n  ulong uVar4;\n  int iVar5;\n  \n  sVar2 = strlen(param_1);\n  uVar4 = sVar2 & 0xffffffff;\n  if ((int)sVar2 != 0) {\n    sVar3 = strlen(param_2);\n    iVar5 = (int)sVar2 - (int)sVar3;\n    if (iVar5 < 0) {\n      uVar4 = 0;\n    }\n    else {\n      uVar4 = 0;\n      pcVar1 = param_1 + (long)iVar5 + 1;\n      do {\n        iVar5 = strncmp(param_1,param_2,(long)(int)sVar3);\n        uVar4 = (ulong)((int)uVar4 + (uint)(iVar5 == 0));\n        param_1 = param_1 + 1;\n      } while (param_1 != pcVar1);\n    }\n  }\n  return uVar4;\n}",
        "optimized_func": "#include <string.h> // Required for strlen and strncmp\n#include <stddef.h> // Required for size_t\n\n// The original 'ulong' type is mapped to 'unsigned long' for standard C compatibility.\n// Parameters are made 'const char*' as the strings are read, not modified.\nunsigned long func0(const char *param_1, const char *param_2)\n{\n    size_t len1 = strlen(param_1);\n    size_t len2 = strlen(param_2);\n    unsigned long count = 0; // Corresponds to the original 'uVar4'\n\n    // The original code returns 0 if param_1 is an empty string.\n    // This check explicitly preserves that behavior.\n    if (len1 == 0) {\n        return 0;\n    }\n\n    // The original code proceeds only if param_1 is not shorter than param_2.\n    // If len1 < len2, the loop below will not execute, and 'count' (which is 0) will be returned,\n    // matching the original behavior (where 'uVar4' would be set to 0 and returned).\n    if (len1 >= len2) {\n        // This loop iterates through all possible starting positions of 'param_2' within 'param_1'.\n        // The loop runs from index 0 up to (len1 - len2), inclusive.\n        // This replaces the 'do-while' loop and eliminates the 'pcVar1' intermediate variable.\n        for (size_t i = 0; i <= len1 - len2; ++i) {\n            // Checks if the substring of 'param_1' starting at current 'i' matches 'param_2'.\n            // The result of strncmp (0 for match) is used directly, eliminating an 'iVar5' usage.\n            if (strncmp(param_1 + i, param_2, len2) == 0) {\n                count++; // Increments count if a match is found.\n            }\n        }\n    }\n\n    // The behavior for an empty 'param_2' (len2 == 0) and non-empty 'param_1' (len1 > 0)\n    // is preserved: 'count' will be 'len1 + 1'.\n    return count;\n}",
        "optimization_status": true
  """
  # Load the JSON file
  with open(json_file_path, "r") as f:
    dataset = json.load(f)
    
  # this json file contains the entire dataset, so iterate through it
  for data in dataset:
    if data["optimization_status"] != True:
      continue  # skip unsuccessful optimizations
    
    print(f"Processing function index: {data['index']}")
    total += 1
    
    # get the includes from data["optimized_func"] and data["test"]
    c_include = data["func_dep"] + "\n"
    c_optimized = data["optimized_func"]
    c_test = data["test"]
    for line in data["optimized_func"].splitlines():
      if "include" in line:
        c_include += line + "\n"
        c_optimized = c_optimized.replace(line,"")
    for line in data["test"].splitlines():
      if "include" in line:
        c_include += line + "\n"
        c_test = c_test.replace(line,"")
    # add the 'using namespace std'
    if data["language"] == "cpp":
      c_include += "using namespace std;\n"
        
    original_c_code = c_include + "\n" + c_optimized + "\n" + c_test
    #print(f"ORIGINAL C CODE : {original_c_code}")
    language = data["language"]
    
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_dir_path = Path(temp_dir)
      c_file_path = temp_dir_path / f"temp_code.{'cpp' if language == 'cpp' else 'c'}"
      with open(c_file_path, "w") as f:
        f.write(original_c_code)
      
      # attempt to compile and execute the original code
      compiled, executed, runtime_message = compile_and_execute(c_file_path, language)
      if not compiled:
        print("Compilation Error : ", runtime_message)
        c_fail += 1
      elif compiled and executed:
        ce_success += 1
        print(f"EXE Rate: {ce_success}/{total} ({ce_success/total*100:.2f}%)")
      else:
        e_fail += 1
        print("Error : ", runtime_message)
        
        
    print(f"Compilation failures: {c_fail}\n Execution failures: {e_fail}\n Successful executions: {ce_success} out of {total}\n")
      
    
  return ce_success, total
    
    
def main():
  """
  Main function to process all JSON files in the corpus root directory.
  """
  
  json_files = list(static_repaired_root.glob("*.json"))
  print(f"Found {len(json_files)} JSON files to process.")
  
  for json_file in json_files:
    print(f"Processing file: {json_file.name}")
    success, total = process_json_file(json_file)
        


if __name__ == "__main__":
  main()