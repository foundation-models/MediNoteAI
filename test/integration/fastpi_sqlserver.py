import argparse
import os
import pyodbc
from fastapi import FastAPI
import uvicorn

app = FastAPI()

db_connection_templae = """
    DRIVER=ODBC Driver 18 for SQL Server;
    SERVER={DB_SERVER};
    DATABASE={DB_NAME};
    UID={DB_USERNAME};
    PWD={DB_PASSWORD};
    TrustServerCertificate=yes;
    Encrypt=yes;
    """  

# Create the connection string
conn_str = db_connection_templae.format(
    DB_SERVER=os.getenv("DB_SERVER"),
    DB_NAME=os.getenv("DB_NAME"),
    DB_USERNAME=os.getenv("DB_USERNAME"),
    DB_PASSWORD=os.getenv("DB_PASSWORD"),
)
print(f"Connection string: {conn_str}")
    

@app.get("/")
async def root():
    # Establish the connection
    try:
        conn = pyodbc.connect(conn_str)
        return {"message": "Connection successful!"}
    except Exception as e:
        return {"error": e.args[1]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
