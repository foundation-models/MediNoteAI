import argparse
from flask import Flask, request, send_file
from flask_cors import CORS
from pandas import read_parquet
import os

# from medinote.augmentation.merger import (
#     merge_all_screened_files,
#     merge_all_sqlcoder_files,
# )
# from medinote.augmentation.sqlcoder import parallel_generate_synthetic_data
# from medinote.curation.screening import api_screening

from medinote import initialize

config, logger = initialize()

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def default():
    """
    Returns a 200 to signal that the server is up.
    """
    return {"status": "ok"}

@app.route("/liveness", methods=["GET"])
def liveness():
    """
    Returns a 200 to signal that the server is up.
    """
    return {"status": "live"}

@app.route("/readiness", methods=["GET"])
def readiness():
    """
    Returns a 200 if the server is ready for prediction.
    """
    return {"status": "ready"}

@app.route("/question2api", methods=["POST"])
def upload_question():
    file = request.files['file']
    # Read the uploaded file
    contents = file.read()
    # Save the contents as a file
    given_schema = config.schemas.get("deal")
    input_path = config.sqlcoder.get("input_path")
    with open(input_path, "wb") as f:
        f.write(contents)
    #parallel_generate_synthetic_data("deal", given_schema=given_schema)
    # merge_all_sqlcoder_files()
    # api_screening()
    # merge_all_screened_files()

    output_path = config.screening.get("merge_output_path")

    df = read_parquet(output_path)
    # Convert the DataFrame to a CSV file
    csv_content = df.to_csv(index=False)

    # Check how many rows of df on column screening has 'totalRecords' in it
    total_records = df[df['screening'].str.contains('totalRecords')].shape[0]

    # Check how many rows of df on column screening has 'totalRecords' but doesn't have '0.0' in it
    total_records_without_zero = df[df['screening'].str.contains('totalRecords') & ~df['screening'].str.contains('0.0')].shape[0]

    # Add a comment to csv_content
    csv_content = f"# {df.shape[0]} records successfully processed. {total_records} passed via api and {total_records_without_zero} returned results \\n" + csv_content

    # Save the CSV content to a file
    output_file_path = os.path.join(os.getcwd(), 'output_file.csv')
    with open(output_file_path, 'w') as f:
        f.write(csv_content)

    # Return the CSV file for the user to download
    return send_file(output_file_path, as_attachment=True, attachment_filename='output_file.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)