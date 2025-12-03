import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import glob
import os
import io
import zstandard as zstd
from ..utils.exebench import Wrapper, diff_io, exebench_dict_to_dict
from pathlib import Path
import yaml


# Config.yaml paths
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
print(f"Loading config from: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

static_repaired_root = Path(config["paths"]["exebench_static_repair_path"])

def verify_exe(row,c_func_optimized):
  
  all_ok = True
  for i in range(len(row['io_pairs'])):
    print('Input:', row['io_pairs'][i]['input'])
    print('Output:', row['io_pairs'][i]['output'])
  
    synth_wrapper = Wrapper(c_deps=row['func_dep'] + '\n' + (row['io_pairs'][i]['dummy_funcs'] or '') + '\n' + c_func_optimized,
                                func_c_signature=row['func_head_types'].replace('extern', ''), func_assembly=None,
                                cpp_wrapper=row['test'])
    observed_output = synth_wrapper(exebench_dict_to_dict(row['io_pairs'][i]['input']))
    
    ok = diff_io(observed_output=observed_output,
                    expected_output=exebench_dict_to_dict(row['io_pairs'][i]['output']))
    if not ok:
      all_ok = False
      break
    print('Test case', i, 'passed.')
    
  return all_ok


def process_json_file(path: str):
  total = 0
  passed = 0
  with open(path, 'r') as f:
    dataset = json.load(f)
  
  # iterate through each data point and verify executability with optimized code
  for data in dataset:
    total += 1
    if data["optimization_status"] != True:
      continue  # skip unsuccessful optimizations
    print(f"Processing function index: {data['index']}")
    c_func_optimized = data["optimized_func"]
    try:
      pass_execution = verify_exe(data, c_func_optimized)
      if pass_execution:
        passed += 1
        print(f"Function index {data['index']} passed execution tests.")
      else:
        print(f"Function index {data['index']} failed execution tests.")
    except Exception as e:
      print(f"Error processing function index {data['index']}: {str(e)}")
    
    print(f"Total functions processed: {total}")
    print(f"Total functions passed: {passed}")
    print(f"Pass rate: {passed / total * 100:.2f}%")



      
if __name__ == "__main__":
  json_file_path = Path(static_repaired_root) / "exebench_data.json"
  process_json_file(json_file_path)
    


