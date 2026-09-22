import os
import time
from typing import Optional
from abstract_essentials import (
    get_caller_dir,
    safe_dump_to_file,
    safe_load_from_json,
    write_to_file,
)
from hugpy_platform.utils import get_messages, get_request_id


def postRequest(url: str, data=None, timeout: float = 600.0):
    """POST ``data`` as JSON and return the decoded body (legacy deepcoder helper)."""
    import requests

    resp = requests.post(url, json=data, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        return resp.text
# NOTE: this is the legacy hosted *deepcoder* codegen endpoint, NOT the hugpy
# central (that's HUGPY_BASE_URL → abstract_hugpy_dev.central). Kept separate on
# purpose; de-hardcoded so a dead domain isn't pinned in source. Override with
# HUGPY_DEEPCODER_URL.
HUGPY_URL = os.environ.get("HUGPY_DEEPCODER_URL", "https://hugpy.abstractendeavors.com").rstrip("/")
DEEPCODER_URL = f"{HUGPY_URL}/deepcoder"
DEEPCODER_GENERATE_URL = f"{DEEPCODER_URL}/generate_unbounded"
def get_responses_dir() -> str:
    abs_dir = get_caller_dir()
    return os.path.join(abs_dir, "responses")

def get_response_dir(folder):
    full_dir = get_responses_dir()
    parts = [part for part in folder.split('/') if part]
    for part in parts:
        full_dir = os.path.join(full_dir,part)
    return full_dir

def make_response_dir(folder):
    full_dir = get_responses_dir()
    parts = [part for part in folder.split('/') if part]
    for part in parts:
        full_dir = os.path.join(full_dir,part)
        os.makedirs(full_dir,exist_ok=True)
    return full_dir

def get_time_response_dir(folder):
    response_dir = get_response_dir(folder)
    return os.path.join(response_dir,str(time.time()))

def make_time_response_dir(folder):
    response_dir = make_response_dir(folder)
    time_dir = os.path.join(response_dir,str(time.time()))
    os.makedirs(time_dir,exist_ok=True)
    return time_dir

def get_response_path(folder):
    time_response_dir = get_time_response_dir(folder)
    return os.path.join(time_response_dir,'response.json')

def make_response_path(folder):
    time_response_dir = make_time_response_dir(folder)
    return os.path.join(time_response_dir,'response.json')
    
def save_response(response,folder):
    response_path = make_response_path(folder)
    full_dir = os.path.dirname(response_path)
    safe_dump_to_file(data=response,file_path=response_path)
    get_code_from_respose(response,code="sh",directory=full_dir)
    get_code_from_respose(response,code="bash",directory=full_dir)
    get_code_from_respose(response,code="python",directory=full_dir)
    return full_dir

def call_and_code(model_config: dict, config_json: Optional[str] = None,directory=None) -> Optional[str]:
    prompt = get_prompt(model_config, config_json=config_json)
    response = call_generate(prompt)
    full_dir = save_response(response,model_config.get('folder'))
    return full_dir
# ---------------------------------------------------------------------------
# Code-block extraction (was get_py_code, with the recursion bug fixed)
# ---------------------------------------------------------------------------

def get_code(text:str or dict,code: str) -> Optional[str]:
    """Pull the first ```python ... ``` block out of a model response.

    Returns None if the input isn't a string or has no python fence. The
    earlier recursive version rebound `values` to itself on every dict/list
    branch, which loops; this just does the one job it needs to do.
    """
    text = str(text)
    if not isinstance(text, str) or f"```{code}" not in text:
        return None
    return str(text).split(f"```{code}", 1)[1].split("```", 1)[0].strip()
def get_all_code(text:str or dict,code: str) -> Optional[list]:
    text = str(text)
    prefix = f"```{code}"
    all_parts = []
    if isinstance(text, str) and prefix in text:
        for part in text.split(prefix)[1:]:
            all_parts.append(str(part).split("```", 1)[0].strip())
    return all_parts
def call_generate(prompt: str, model_key: str = "qwen3_coder_next_gguf",
                  max_new_tokens: int = 4050) -> dict:
    data = {
        "messages": get_messages(prompt),
        "request_id": get_request_id(),
        "model_key": model_key,
        "max_new_tokens": max_new_tokens,
    }
    return postRequest(url=DEEPCODER_GENERATE_URL, data=data)

# ---------------------------------------------------------------------------
# Prompt building + codegen orchestration (unchanged behavior, tightened types)
# ---------------------------------------------------------------------------

def get_prompt(model_config: dict, config_json: Optional[str] = None) -> str:
    data = None
    if config_json:
        data = safe_load_from_json(config_json)

    prompt = (
        f"please provide a python code to run the following huggingface "
        f"local model:\nmodel_config: {model_config}\n\n"
    )
    if data:
        prompt += f"config.json data:{data}"
    return prompt

def get_code_from_respose(response,code=None,directory=None):
    if not code:
        return
    code_js = {"python":'.py','sh':'.sh','bash':".sh"}
    if not directory:
        return 

    ext = code_js.get(code)
    
    if not ext:
        return
    code_dir = os.path.join(directory,code)
    os.makedirs(code_dir,exist_ok=True)
    code_scripts = get_all_code(response,code)
    for i,code_script in enumerate(code_scripts):
        basename = f"{code}_{i}{ext}"
        code_path = os.path.join(code_dir,basename)
        if code_script:
            write_to_file(file_path=code_path,contents=code_script)
