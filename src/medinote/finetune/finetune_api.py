import argparse
import logging
from multiprocessing.pool import ThreadPool
import os

import requests
from medinote.finetune.apply_lora import apply_lora_create_new_model

import uvicorn

from fastapi import Body, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from multiprocessing import set_start_method
from fastchat.serve.base_model_worker import app
from medinote.finetune.finetune_lora import train

current_path = os.path.dirname(__file__)

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)
logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

callback_url = os.environ.get("CALLBACK_URL")

set_start_method(
    "spawn", force=True
)  # needed for multiprocessing otherwise getting Cannot re-initialize CUDA in forked subprocess... error


# make this only for dev environment but for production list the allowed origins
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def success_call(
    task_id: str = None,
    body: dict = {},
):
    apply_lora_create_new_model(
        base_model_path=body["model_name_or_path"],
        lora_path=body["output_dir"],
        target_model_path=body['target_model_path'],
        trust_remote_code=True,
    )
    logger.info(f"Calling callback url: {callback_url}/trainings/train/end/succeed", flush=True)
    response = requests.get(url=f'{callback_url}/trainings/train/end/succeed')
    response.raise_for_status()

# error callback function
def fail_call(error: Exception, task_id: str = None):
    logger.error(f"Error: {error}", flush=True)
    logger.error(f"Calling callback url: {callback_url}/trainings/train/end/failed", flush=True)
    response = requests.get(url=f'{callback_url}/trainings/train/end/failed')
    response.raise_for_status()


def train_wrapper(body: dict = None, task_id: str = None):
    try:
        train(body)
        if task_id:
            success_call(
                task_id=task_id,
                body=body,
            )
    except Exception as e:
        if task_id:
            fail_call(task_id=task_id, error=e)
        # raise Exception('Lora').with_traceback(e.__traceback__)
        raise e
 
# function for initializing the worker thread
def initializer_worker():
    # raises an exception
    # raise Exception('Something bad happened!')
    pass

@app.post("/train/{task_id}")
async def train_api(
    body: dict = Body(..., media_type="application/json"),
    task_id: str = Path(title="task_id"),
):
    try:
        model_dir = os.environ.get("model_dir")
        if "model_name_or_path" in body:
            body["model_name_or_path"] = os.path.join(
                model_dir, body["model_name_or_path"]
            ) if model_dir else body["model_name_or_path"]
        datasets_dir = os.environ.get("datasets_dir")
        if "data_path" in body:
            body["data_path"] = os.path.join(datasets_dir, body["data_path"]) if datasets_dir else body["data_path"]
        if "eval_data_path" in body:
            body["eval_data_path"] = os.path.join(datasets_dir, body["eval_data_path"]) if datasets_dir else body["eval_data_path"]
        body["output_dir"] = f"{body['model_name_or_path']}_lora/{task_id}"
        # with ThreadPoolExecutor(max_workers=2, initializer=initializer_worker) as executor:
        #     executor.submit(train_wrapper, body, task_id)

        pool = ThreadPool()
        pool.apply_async(func=train_wrapper, args=(body, task_id))

        # # with Pool() as pool:
        #     # issue tasks to the process pool
        #     result = pool.apply_async(func=train_wrapper, args=(body,task_id))
        #     # # close the process pool
        #     # pool.close()
        #     # # wait for all tasks to complete
        #     # pool.join()
        #     result.get()
        return f"Training process started. You will receive a callback with id {task_id} when it is finished."
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=f"Train was not successful due to error: -- {e.args[0]} --")


@app.post("/merge_peft_model_parts")
async def merge_peft_model_parts_api(
    body: dict = Body(..., media_type="application/json")
):
    try:
        apply_lora(
            base_model_path=body["base_model_path_or_name"],
            lora_path=body["lora_model_path_or_name"],
            target_model_path=body["merged_model_path"],
        )
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=e.args[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    app.title = "FineTuning API"
    app.description = "API for FineTuning"
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
