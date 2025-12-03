import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from enum import Enum
import argparse
import shutil
import yaml
from .ghidra_parser import *


# Config.yaml paths
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
print(f"Loading config from: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    


class Ghidra:
  def __init__(self, ghidra_path: str = config["paths"]["ghidra_path"], post_script: str = config["paths"]["postscript_path"]):
    self.ghidra_path = ghidra_path
    self.post_script = post_script
    

  def convert_executable_to_ghidra(
    self,
    executable_path: str,
    output_dir: str) -> List[Tuple[bool, str]]:
    """
    Convert an executable binary to Ghidra decompiled pseudo C code.
    Args:
        executable_path: Path to the executable binary
        output_dir: Directory to save the decompiled output
    Returns:
        List of tuples (success, output_path or error_message)
    """
    print(f"Starting Ghidra decompilation for {executable_path}")
    executable_path = Path(executable_path)
    results = []
    
    if not executable_path.exists():
      msg = f"Executable not found: {executable_path}"
      results.append((False, msg))
      return results
    
    with tempfile.TemporaryDirectory() as temp_dir:
      pid = os.getpid()
      output_file = Path(temp_dir) / f"{executable_path.stem}_decompiled.c"
      command = [
        self.ghidra_path,
        temp_dir,
        "tmp_ghidra_proj",
        "-import", executable_path,
        "-postScript", self.post_script, output_file,
        "-deleteProject",
      ]
      try:
        subprocess.run(command, text=True, capture_output=True, check=True, timeout=120)
        print(f"Ghidra decompilation succeeded for {executable_path.name}")
        with open(output_file, 'r') as f:  
          #enriched_code = parse_and_enrich(output_file)
          enriched_code = f.read()
        return True, enriched_code
      except Exception as e:
        print(f"Ghidra decompilation failed: {e}")
        return False, str(e)
