import logging
import os
import tempfile
from typing import Any, Dict, Optional

import requests


from hugpy_media.whisper_model.src.model.model import get_whisper_model

logger = logging.getLogger(__name__)
def extension_from_content_type(content_type: str) -> str:
    content_type = (content_type or "").lower().split(";")[0].strip()

    mapping = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mp4": ".m4a",
        "audio/aac": ".aac",
        "audio/ogg": ".ogg",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
    }

    return mapping.get(content_type, ".media")

def stream_url_to_temp_file(
    url: str,
    session: Optional[requests.Session] = None,
    timeout: int = 30,
    chunk_size: int = 1024 * 1024,
    suffix: Optional[str] = None,
    max_bytes: Optional[int] = None,
) -> str:    
    """
    Stream a remote media URL to a temporary file.

    This avoids holding the whole file in memory.

    Args:
        url:
            Direct audio/video URL.
        session:
            Optional requests.Session.
        timeout:
            Request timeout.
        chunk_size:
            HTTP chunk size.
        suffix:
            Optional file suffix, e.g. ".mp3", ".wav", ".mp4".
        max_bytes:
            Optional safety limit.

    Returns:
        Temporary file path.
    """
    from abstract_webtools import derive_approved_headers_user_agent_session_for_url
    owns_session = session is None
    if not session:
        user_agent,headers,session,source_code = derive_approved_headers_user_agent_session_for_url(url)
        session.header.update(headers)
    downloaded = 0
    temp_path = None

    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            
            if suffix is None:
                suffix = extension_from_content_type(
                    r.headers.get("content-type", "")
                )

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as tmp:
                temp_path = tmp.name

                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue

                    downloaded += len(chunk)

                    if max_bytes is not None and downloaded > max_bytes:
                        raise ValueError(
                            f"Download exceeded max_bytes limit: {max_bytes}"
                        )

                    tmp.write(chunk)

        if not temp_path or not os.path.isfile(temp_path):
            raise FileNotFoundError("Temporary media file was not created.")

        logger.info(f"Streamed URL to temporary file: {temp_path}")
        return temp_path

    except Exception:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    finally:
        if owns_session:
            session.close()

def whisper_transcribe_url_stream(
    url: str,
    model_size: str = "small",
    language: str = "english",
    task: Optional[str] = None,
    whisper_model_path: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout: int = 30,
    chunk_size: int = 1024 * 1024,
    keep_temp: bool = False,
    max_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Stream a media URL with requests, save it temporarily, then transcribe it.

    This works best for direct URLs like:
        .mp3
        .wav
        .m4a
        .mp4
        .webm

    It will not reliably work for YouTube page URLs unless the URL is already
    a direct media stream.
    """

    temp_path = stream_url_to_temp_file(
        url=url,
        session=session,
        timeout=timeout,
        chunk_size=chunk_size,
        max_bytes=max_bytes,
    )

    try:
        model = get_whisper_model(
            module_size=model_size,
            whisper_model_path=whisper_model_path,
        )

        options: Dict[str, Any] = {
            "language": language,
        }

        if task:
            options["task"] = task

        result = model.transcribe(temp_path, **options)

        result["source_url"] = url
        result["temp_path"] = temp_path if keep_temp else None

        return result

    finally:
        if not keep_temp and os.path.exists(temp_path):
            os.remove(temp_path)


