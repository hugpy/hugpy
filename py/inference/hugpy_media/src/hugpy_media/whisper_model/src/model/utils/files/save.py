import json
import os
from typing import Any
def save_transcript_outputs(
    whisper_result: dict[str, Any],
    transcript_json_path: str,
    transcript_text_path: str,
) -> None:
    os.makedirs(os.path.dirname(transcript_json_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(transcript_text_path) or ".", exist_ok=True)

    with open(transcript_json_path, "w", encoding="utf-8") as file:
        json.dump(whisper_result, file, indent=2, ensure_ascii=False)

    text = whisper_result.get("text", "")
    with open(transcript_text_path, "w", encoding="utf-8") as file:
        file.write(text.strip() + "\n")
