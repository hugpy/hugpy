from enum import Enum

class Runtime(str, Enum):
    transformers = "transformers"
    gguf = "gguf"                 # HF Hub's library tag for GGUF repos
    dataset = "dataset"
    unknown = "unknown"


