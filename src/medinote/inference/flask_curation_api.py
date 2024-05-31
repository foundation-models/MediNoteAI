from glob import glob
import io
import json
import os
import time
from flask import Flask, jsonify, redirect, render_template, render_template_string, request, send_file, make_response
from pathlib import Path

from pandas import read_parquet
import yaml
from medinote.augmentation.sqlcoder import parallel_generate_synthetic_data
from medinote.curation.screening import api_screening
from medinote.augmentation.merger import (
    merge_all_screened_files,
    merge_all_sqlcoder_files,
)

from medinote import initialize
from medinote.inference.inference_prompt_generator import row_infer


config, logger = initialize()

app = Flask(__name__, template_folder='templates') 

is_rpu: bool = True

try:
    with open(
        "/home/agent/workspace/query2sql2api/config/config.yaml", "r"
    ) as file:
        conf = yaml.safe_load(file)
        fs_conf = conf.get('fs')
        rpu_conf = conf.get('rpu')
        if not fs_conf or not rpu_conf:
            raise Exception("Chat config not found")  
  

    def get_response(userText: str):
        row={"query": userText}
        row = row_infer(row=row, config=rpu_conf) if is_rpu else row_infer(row=row, config=fs_conf)
        status_code = row['status_code']
        if status_code == 200 and "inference_response" in row:
            generated_sql = row["inference_response"]
            try:
                api_response = json.loads(row['api_response'])
                if 'datarows' in api_response:
                    datarows = api_response['datarows']
                    
                    data = []
                    if is_rpu:
                        for datarow in datarows:
                            data.append({"id": datarow[23], 'name': datarow[6], 'description': datarow[35]})
                    else:
                        for datarow in datarows:
                            data.append({"id": datarow[36], 'name': datarow[6], 'description': datarow[35]})
                    if data:
                        html_content = render_template('table.html', data=data)
                        # Render the HTML content
                        api_response_in_html = render_template_string(html_content)
                        return  f"Your question converted to <br/>{generated_sql} <br/>and DataCortex response is <br/>{api_response_in_html}"
            except Exception as e:
                return f"Sorry, I received the following error: <br/>{repr(e)}."
        else:
            return f"Sorry, I received the following error: <br/>{row.get('inference_response')}."
        return  f"Your question converted to <br/>{generated_sql}"
except Exception as e:
    logger.warning(f"Chat Service is not available {repr(e)}")
    



@app.route('/', methods=['GET'])
def root():
    """
    Returns a 200 to signal that the server is up.
    """
    return {"status": "ok"}   


@app.route("/liveness")
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
    return {"status": "live"}
    
@app.route("/question2api")
def get_upload_form():
    # Assuming 'config' and 'inference' are already defined and imported
    flask_curations_url = config.get("inference").get("flask_curations_url")
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
    given_schema = config.get("schemas").get("dealcloud_provider_fs_companies_a")
    input_path = config.get("sqlcoder").get("input_path")
    with open(input_path, "wb") as f:
        f.write(contents)


    for f in glob("/tmp/*.parquet"):
        os.remove(f)
    # Your processing code
    parallel_generate_synthetic_data("deal", given_schema=given_schema)
    merge_all_sqlcoder_files()
    api_screening()
    merge_all_screened_files()

    output_path = config.get("screening").get("merge_output_path")
    
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

    
@app.route("/fs/chat") 
def fs_chat():
    global is_rpu
    is_rpu = False
    return render_template("index.html")


    
@app.route("/rpu/chat") 
def rpu_chat():
    global is_rpu
    is_rpu = True
    return render_template("index.html")


@app.route("/get")
# Function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    return str(get_response(userText))

@app.route('/refresh')
def refresh():
    time.sleep(600) # Wait for 10 minutes
    return redirect('/refresh')



if __name__ == "__main__":
    # app.run(debug=False, host='0.0.0.0', port=8888, threaded=False, processes=1)
    app.run(port=8888,host='0.0.0.0')