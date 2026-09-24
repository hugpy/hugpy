import logging

logger = logging.getLogger(__name__)
from hugpy_platform.constants import MODELS_DIR
from hugpy_server.app.functions.imports.utils.constants import get_hf_api


def model_size(hub_id: str) -> int | None:
    """Total repo bytes via the PERMANENT central HF metadata cache (replaced
    the old in-memory 600s-TTL dict): first ask per repo ever hits HF, every
    later ask is served from SQLite — fetch-once, no expiry (operator policy;
    see comms/model_metadata.py). A live-call failure logs and returns None,
    exactly the old contract."""
    from hugpy_storage.model_metadata import fetch_repo_info, sum_sibling_sizes
    try:
        payload = fetch_repo_info(hub_id, files_metadata=True, api=get_hf_api(),
                                  purpose="discovery")
    except Exception as exc:
        logger.warning("model_size(%s) failed: %s", hub_id, exc)   # don't hide it
        return None
    return sum_sibling_sizes(payload)

def free_bytes() -> int | None:
    from hugpy_platform.filesystem import free_bytes as shared_free_bytes
    return shared_free_bytes(MODELS_DIR)
