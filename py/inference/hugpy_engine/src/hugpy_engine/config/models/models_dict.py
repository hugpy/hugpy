MODELS={ "Falconsai-text-summarization": {
    "model_max_length": 512,
"include": None,
"name": "Falconsai-text-summarization",
"framework": "transformers",
"hub_id": "Falconsai/text-summarization",
"filename": None,
"folder": "Falconsai/text-summarization",
"tasks": ["text-summarization"],
"primary_task": "text-summarization",
"port": None
  },
"led-large-16384": {
"model_max_length": 16384,
"include": None,
"name": "led-large-16384",
"framework": "transformers",
"hub_id": "allenai/led-large-16384",
"filename": None,
"folder": "allenai/led-large-16384",
"tasks": ["text-summarization"],
"primary_task": "text-summarization",
"port": None
  },
"flan-t5-xl": {
"model_max_length": 1024,
"include": None,
"name": "flan-t5-xl",
"framework": "transformers",
"hub_id": "google/flan-t5-xl",
"filename": None,
"folder": "google/flan-t5-xl",
"tasks": ["text-summarization", "text2text-generation"],
"primary_task": "text-summarization",
"port": None
  },
"all-minilm-l6-v2": {
"model_max_length": 512,
"include": None,
"name": "all-minilm-l6-v2",
"framework": "transformers",
"hub_id": "sentence-transformers/all-minilm-l6-v2",
"filename": None,
"folder": "sentence-transformers/all-minilm-l6-v2",
"tasks": ["feature-extraction", "sentence-similarity"],
"primary_task": "feature-extraction",
"port": None
  },
"gte-large-en-v1.5": {
"model_max_length": 8192,
"include": None,
"name": "gte-large-en-v1.5",
"framework": "transformers",
"hub_id": "Alibaba-NLP/gte-large-en-v1.5",
"filename": None,
"folder": "Alibaba-NLP/gte-large-en-v1.5",
"tasks": ["feature-extraction", "sentence-similarity"],
"primary_task": "feature-extraction",
"port": None
  },
"whisper-large-v3": {
"model_max_length": 448,
"include": None,
"name": "whisper-large-v3",
"framework": "transformers",
"hub_id": "openai/whisper-large-v3",
"filename": None,
"folder": "openai/whisper-large-v3",
"tasks": ["automatic-speech-recognition"],
"primary_task": "automatic-speech-recognition",
"port": None
  }
}
