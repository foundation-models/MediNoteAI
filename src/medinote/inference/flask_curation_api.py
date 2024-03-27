from glob import glob
import io
import os
from flask import Flask, request, send_file, make_response
from pathlib import Path

from pandas import read_parquet
from medinote.augmentation.sqlcoder import parallel_generate_synthetic_data
from medinote.curation.screening import api_screening
from medinote.augmentation.merger import (
    merge_all_screened_files,
    merge_all_sqlcoder_files,
)

from medinote import initialize


config, logger = initialize()

app = Flask(__name__)


@app.route('/', methods=['GET'])
def root():
    """
    Returns a 200 to signal that the server is up.
    """
    return {"status": "ok"}   

@app.route("/question2api")
def get_upload_form():
    # Assuming 'config' and 'inference' are already defined and imported
    flask_curations_url = config.inference.get("flask_curations_url")
    form_html = Path("/home/agent/workspace/MediNoteAI/src/medinote/inference/upload_form.html").read_text()
    
    # Get the host from the request headers
    host = request.headers.get('Host', 'localhost:8888')
    flask_curations_url = flask_curations_url or f'http://{host}'
    
    # Replace placeholders in the HTML
    form_html = form_html.replace("http://localhost:8888", flask_curations_url).replace('service_name', 'question2api')
    
    # Create and return a response
    response = make_response(form_html)
    response.headers['Content-Type'] = 'text/html'
    return response


@app.route("/question2api/", methods=["POST"])
def upload_question():
    # Receive the file
    file = request.files['file']

    # Read the uploaded file
    contents = file.read()

    # Save the contents as a file
    given_schema = config.schemas.get("dealcloud_provider_fs_companies_a")
    input_path = config.sqlcoder.get("input_path")
    with open(input_path, "wb") as f:
        f.write(contents)


    for f in glob("/tmp/*.parquet"):
        os.remove(f)
    # Your processing code
    parallel_generate_synthetic_data("deal", given_schema=given_schema)
    merge_all_sqlcoder_files()
    api_screening()
    merge_all_screened_files()

    output_path = config.screening.get("merge_output_path")
    
    df = read_parquet(output_path)
    
    # Check how many rows of df on column screening has 'totalRecords' in it
    df['api_passed'] = df['screening'].str.contains('totalRecords')
    total_records = df[df['api_passed']].shape[0]
    
    df['api_passed_return_results'] = df['screening'].str.contains('totalRecords') & ~df['screening'].str.contains(': 0')
    # Check how many rows of df on column screening has 'totalRecords' but doesn't have '0.0' in it
    total_records_without_zero = df[df['api_passed_return_results']].shape[0]
    
    df = df[['text', 'modified_sql', 'api_passed', 'api_passed_return_results']]
    # Convert the DataFrame to a CSV file
    csv_content = df.to_csv(index=False)
    # Add a comment to csv_content
    csv_content = f"# {df.shape[0]} records successfully processed. {total_records} passed via api and {total_records_without_zero} returned results \n" + csv_content


    # Create a response
    response = send_file(
        io.BytesIO(csv_content.encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="output.csv"
    )
    return response

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8888, threaded=False, processes=1)