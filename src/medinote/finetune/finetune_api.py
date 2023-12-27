import argparse
import logging
from multiprocessing.pool import ThreadPool
import os
from fastchat.model import apply_lora

import uvicorn

from fastapi import Body, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from multiprocessing import set_start_method
from fastchat.serve.base_model_worker import app
from medinote.finetune.finetune_lora import train

set_start_method(
    "spawn", force=True
)  # needed for multiprocessing otherwise getting Cannot re-initialize CUDA in forked subprocess... error

current_path = os.path.dirname(__file__)

log = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


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
    base_model_path_or_name: str = None,
    lora_model_path_or_name: str = None,
    full_model_dir: str = None,
):
    # call success_call(task_id)
    pass


# error callback function
def fail_call(error: Exception, task_id: str = None):
    print(f"Error: {error}", flush=True)
    # call failed_call(task_id, error)


def train_wrapper(body: dict = None, task_id: str = None):
    try:
        train(body)
        if task_id:
            success_call(
                task_id=task_id,
                base_model_path_or_name=body["base_model_path_or_name"],
                lora_model_path_or_name=body["output_dir"],
                full_model_dir=f"{body['full_model_dir']}/{task_id}",
            )
    except Exception as e:
        if task_id:
            fail_call(task_id=task_id, error=e)
        raise e


@app.post("/train/{task_id}")
async def train_api(
    body: dict = Body(..., media_type="application/json"),
    task_id: str = Path(title="task_id"),
):
    try:
        model_dir = os.environ["model_dir"]
        if "model_name_or_path" in body:
            body["model_name_or_path"] = os.path.join(
                model_dir, body["model_name_or_path"]
            )
        datasets_dir = os.environ["datasets_dir"]
        if "data_path" in body:
            body["data_path"] = os.path.join(datasets_dir, body["data_path"])
        if "eval_data_path" in body:
            body["eval_data_path"] = os.path.join(datasets_dir, body["eval_data_path"])
        body["output_dir"] = f"{os.environ['output_dir']}/{task_id}"
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
        logging.error(e)
        raise HTTPException(status_code=500, detail=e.args[0])


@app.post("/merge_peft_model_parts")
async def merge_peft_model_parts_api(
    body: dict = Body(..., media_type="application/json")
):
    try:
        apply_lora(
            base_model_path=body["base_model_path_or_name"],
            lora_path=body["lora_model_path_or_name"],
            target_model_path=body["full_model_dir"],
        )
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=e.args[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    app.title = "FineTuning API"
    app.description = "API for FineTuning"
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
