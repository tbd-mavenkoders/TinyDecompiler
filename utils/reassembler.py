'''
Converts the json object of a single json object to a C file.
'''

import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple
import shutil
import json
from .compile import Compiler, OptimizationLevel


class Reassembler:
  def __init__(self):
    pass
  
  def reassemble_file(self,file_object):
    '''
    Reassembles a C file from a JSON object.
    Args:
      file_object (dict): The JSON object representing the C file.
    Returns:
      str: The reassembled C file as a string.
    '''
    c_file = ""
    for func in file_object:
      func_name = func.get("func_name", "")
      c_file += f"// Function: {func_name}\n"
      c_file += func.get("decompiled_code", "") + "\n\n"
    return c_file
  
  def reassemble_project(self, json_path: str) -> Tuple[str, Dict[str, str]]:
    '''
    Reassembles a C project from a JSON object.
    Args:
      json_object (dict): The JSON object representing the C project.
    Returns:
      Tuple[str, Dict[str, str]]:
        - temp_dir_path: Path to the temporary directory
        - file_mapping: Dictionary mapping C filenames to their full paths
        - functions: Dict mapping C filenames to their function names
    '''
    json_object = {}
    with open(json_path, 'r', encoding='utf-8') as f:
      json_object = json.load(f)
    # Extract the project data
    project_name = json_object.get("file", "project")
    decompiled_code = json_object.get("decompiled_code", {})
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix=f"{project_name}__")
    temp_dir_path = Path(temp_dir)
    
    output = {}
    
    # Iterate through each C file in the decompiled code
    for c_filename, functions_list in decompiled_code.items():
      c_file_content = self.reassemble_file(functions_list)
      c_file_path = temp_dir_path / c_filename
      with open(c_file_path, 'w', encoding='utf-8') as f:
        f.write(c_file_content)
      output[c_filename] = str(c_file_path)
    
    return str(temp_dir_path), output

  def cleanup_temp_directory(self, temp_dir_path: str) -> None:
    if os.path.exists(temp_dir_path):
      shutil.rmtree(temp_dir_path)
      
      
    
  
