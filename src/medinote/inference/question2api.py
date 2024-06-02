import argparse
from fastapi import Request, FastAPI, File, Response, UploadFile


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path

from pandas import read_parquet
import uvicorn

from medinote.augmentation.merger import (
    merge_all_screened_files,
    merge_all_sqlcoder_files,
)
from medinote.augmentation.sqlcoder import parallel_generate_synthetic_data
from medinote.curation.screening import api_screening


from medinote import initialize


_, logger = initialize()


app = FastAPI(
    title="Question2API", description="Tuen Question API Vall and get the response"
)

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


@app.get("/question2api", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    flask_curations_url = config.get("inference").get("flask_curations_url")
    form_html = Path(
        "/home/agent/workspace/MediNoteAI/src/medinote/inference/upload_form.html"
    ).read_text()
    
    flask_curations_url = flask_curations_url or (f'http://{request.headers["host"]}')
    # Replace the placeholder with the actual domain
    form_html = form_html.replace(
        "http://localhost:8888", flask_curations_url).replace('service_name', 'question2api')
    return HTMLResponse(content=form_html)


@app.post("/question2api/")
async def upload_question(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    # Save the contents as a file
    given_schema = config.get("schemas").get("dealcloud_provider_fs_companies_a")
    input_path = config.get("sqlcoder").get("input_path")
    with open(input_path, "wb") as f:
        f.write(contents)
    parallel_generate_synthetic_data("deal", given_schema=given_schema)
    merge_all_sqlcoder_files()
    api_screening()
    merge_all_screened_files()

    output_path = config.get("screening").get("merge_output_path")
    
    df = read_parquet(output_path)
    # Convert the DataFrame to a CSV file
    csv_content = df.to_csv(index=False)
    
    # Check how many rows of df on column screening has 'totalRecords' in it
    total_records = df[df['screening'].str.contains('totalRecords')].shape[0]
    
    # Check how many rows of df on column screening has 'totalRecords' but doesn't have '0.0' in it
    total_records_without_zero = df[df['screening'].str.contains('totalRecords') & ~df['screening'].str.contains('0.0')].shape[0]
    
    # Add a comment to csv_content
    csv_content = f"# {df.shape[0]} records successfully processed. {total_records} passed via api and {total_records_without_zero} returned results \n" + csv_content

    # Return the CSV file for the user to download
    response = Response(
        content=csv_content.encode(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=output_file.csv"},
    )
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    app.title = "Generate API"
    app.description = "API for Captioning"
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
