##Models in VRAM — allocation each (3)
##🔥 servingQwen~Qwen3-Coder-Next-GGUF3.6 GiB VRAM · ctx 32768🔥 servingJackrong~Qwen3.5-9B-DeepSeek-V4-Flash-GGUF6.3 GiB VRAM · ctx 32768🔥 servingreaperdoesntknow~Qwen3-1.7B-Thinking-Distilhost RAM — not in VRAM
##12.2 GiB attributed + 256.0 MiB unattributed / foreign + 3.0 GiB KV cache / activations = 15.5 GiB used

smi= """...ot990/comfyui/venv/bin/python  50639 378 MiB
...ct_hugpy_dev/venv/bin/python3  88634 256 MiB
.../hugpy-worker/venv/bin/python 94035 4074 MiB
.../hugpy-worker/venv/bin/python 94173 256 MiB
.../hugpy-worker/venv/bin/python 94174 256 MiB
...orage/engine/bin/llama-server 97350 3654 MiB
...orage/engine/bin/llama-server 126140 6468 MiB"""
derive = """Qwen~Qwen3-Coder-Next-GGUFpid 97350 3.6 GiB
Jackrong~Qwen3.5-9B-DeepSeek-V4-Flash-GGUFpid 126140 6.3 GiB
reaperdoesntknow~Qwen3-1.7B-Thinking-Distilpid 94035 1.8 GiB
comfy 50639 378.0 MiB
cuda_context 94173 256.0 MiB
cuda_context 94174 256.0 MiB
/srv/share/projects/hugpy/dev/abstract_hugpy_dev/venv/bin/python3pid 88634 256 MiB"""
def get_infos(all_batche):
    batch ={"GiB":1024,"MiB":1,"all_size":0,"processes":[]}

    for pidss in all_batche.split('\n'):
        process={}
        splits = pidss.split(' ')
        process["unit"] = splits[-1]
        process["unit_size"] = float(splits[-2])
        process["pid"] = splits[-3]
        process["typ"] = splits[-4]
        process["size"] = batch[process["unit"]]*process["unit_size"]
        batch["all_size"] +=process["size"]
        batch["processes"].append(process)
    return batch        
all_batches = [smi,derive]
for i,all_batche in enumerate(all_batches):
    all_batches[i]=get_infos(all_batche)
    input(all_batches)
