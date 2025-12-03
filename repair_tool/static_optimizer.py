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
    
# Create LLM interface
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
  Arguments :
    c_code: The original C code as a string
    summary: The summary of the function's purpose
    language: The programming language (e.g., "c" or "cpp")
  Returns:
    The formatted initial prompt string.
  """
  initial_prompt = config["prompts"]["system_prompt"]
  prompt = f"{initial_prompt}\n\n```Language:{language}\nSummary:{summary}\n{c_code}\n```"
  return prompt

def summarize_function(ghidra_code: str, asm: str, language: str) -> str:
  """
  Summarize the function purpose using LLM.
  Arguments :
    ghidra_code: The C code obtained from Ghidra decompilation
    asm: The assembly code of the function
    language: The programming language (e.g., "c" or "cpp")
  Returns:
    A summary string describing the function's purpose.
  """
  summary_prompt = config["prompts"]["summary_prompt"]
  prompt = f"{summary_prompt}\n\n```Language:{language}\nCode:{ghidra_code}\nASM:{asm}\n```"
  summary = llm_interface.generate(prompt)
  return clean_llm_output(summary)


def get_optimized_code(original_c_code: str,language: str, max_iterations: int) -> str:
  """
  Generate optimized C code using LLM for the given original C code file.
  Arguments :
    original_c_code: The original C code as a string
    language: The programming language (e.g., "c" or "cpp")
    max_iterations: Maximum number of optimization iterations
  Returns:
    A tuple (success: bool, optimized_code: str)
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
    initial_prompt = get_initial_prompt(original_c_code,language)
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
      print(f"Compilation Errors:\n{error_prompt}")
      repair_prompt = f"{config['prompts']['compilation_error']}\n\n```c\nLanguage:{language}\n{optimized_code}\n```\n\nCompilation Errors:\n{error_prompt}\n\nPlease provide the corrected C code."
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
