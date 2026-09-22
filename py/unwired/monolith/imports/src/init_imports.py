from __future__ import annotations
import os,json,re,unicodedata,bs4,urllib,tempfile,copy,uuid,logging,argparse,json,os,re,requests,glob
import os.path as osp
from pydantic import BaseModel, ConfigDict, Field,model_validator
from dataclasses import dataclass, asdict, field, dataclass, fields,MISSING
dataclass_fields = fields
from typing import (
    List,
    Optional,
    Callable,
    Dict,
    Tuple,
    Union,
    Literal
    )
from urllib.parse import (
    urlunparse,
    unquote,
    quote,
    urlparse,
    parse_qs
    )
from collections import Counter
from typing import *
from PyPDF2 import PdfReader
from uuid import uuid1
from pathlib import Path
from .standalone_utils import *
import requests
from datetime import datetime, timezone
from enum import Enum
from huggingface_hub import hf_hub_download, snapshot_download,HfApi
from huggingface_hub.hf_api import RepoFile
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
logger = get_logFile(__name__)

