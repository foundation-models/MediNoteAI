import asyncio
import importlib
import inspect
import os
from contextlib import asynccontextmanager
from http import HTTPStatus

import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest, ErrorResponse)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
import yaml
from medinote.utils.conversion import string2json
from medinote.inference.inference_prompt_generator import row_infer

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
logger = init_logger(__name__)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())

    yield


app = fastapi.FastAPI(lifespan=lifespan)


def parse_args():
    parser = make_arg_parser()
    return parser.parse_args()


# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)
@app.get("/")
async def default():
    """
    Returns a 200 to signal that the server is up.
    """
    return {"status": "ok"}


@app.get("/liveness")
async def liveness():
    """
    Returns a 200 to signal that the server is up.
    """
    return {"status": "live"}


@app.get("/readiness")
async def readiness():
    """
    Returns a 200 if the server is ready for prediction.
    """

    # Currently simply take the first tenant.
    # to decrease chances of loading a not needed model.
    return {"status": "ready"}


@app.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.get("/version")
async def show_version():
    ver = {"version": vllm.__version__}
    return JSONResponse(content=ver)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())

lock = asyncio.Lock()



try:

    with open(
        f"{os.path.dirname(os.path.abspath(__file__))}/../config/config.yaml", "r"
    ) as file:
        conf = yaml.safe_load(file)  
    

    @app.post("/response")
    async def create_response(raw_request: Request):
        async with lock:
        
            params = await raw_request.json()
            config_node = (
                params.get("config_node")
                or raw_request.query_params.get("config_node")
                or os.getenv("config_node")
                or "vllm_openai_api"
            )
            # row = row_infer(row=params, config=conf.get(config_node))
            prompt = params.get("prompt")
            if not prompt:  
                template = conf.get(config_node).get("prompt_template")
                prompt = template.format(**params)
            template = conf.get(config_node).get("payload_template")
            payload = template.format(**{"prompt": prompt})
            payload = payload.replace("\n", "\\n")
            payload = string2json(payload)
            request = ChatCompletionRequest(
                model=payload.get("model"),
                messages=payload.get("messages"),
                max_tokens=payload.get("max_tokens"),
                stop_token_ids=payload.get("stop_token_ids")
            )
            
            generator = await openai_serving_chat.create_chat_completion(
                    request, raw_request)
            if isinstance(generator, ErrorResponse):
                return JSONResponse(content=generator.model_dump(),
                                    status_code=generator.code)
            if request.stream:
                return StreamingResponse(content=generator,
                                        media_type="text/event-stream")
            else:
                return JSONResponse(content=generator.model_dump())
except Exception as e:
    logger.warning("Custom service are not available due to: ", e)

if __name__ == "__main__":
    args = parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := os.environ.get("VLLM_API_KEY") or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    logger.info(f"vLLM API server version {vllm.__version__}")
    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER)
    openai_serving_chat = OpenAIServingChat(engine, served_model_names,
                                            args.response_role,
                                            args.lora_modules,
                                            args.chat_template)
    openai_serving_completion = OpenAIServingCompletion(
        engine, served_model_names, args.lora_modules)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
    