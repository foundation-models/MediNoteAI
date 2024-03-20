import argparse

import torch
from fastchat.constants import SERVER_ERROR_MSG, ErrorCode
from fastapi import Request, FastAPI, File, UploadFile, Form
import requests

import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import io
from fastapi.responses import JSONResponse

import torch
from transformers import AutoProcessor, VipLlavaForConditionalGeneration
from fastapi.responses import HTMLResponse
from pathlib import Path
