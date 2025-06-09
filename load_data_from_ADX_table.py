# %python
# %pip install azure-kusto-data
# %pip install fastapi
# %restart_python
# pip install fastapi uvicorn azure-kusto-data azure-identity pandas python-dotenv

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
import pandas as pd
from typing import List
from typing import Optional

# Constants (replace with your actual values or load via environment variables)
TABLE_NAME = "ImageBlurData_V2" #"ImageBlurData" # "blur_predictions"
CLUSTER = "https://abc.windows.net" # os.getenv("ADX_CLUSTER", "<your_cluster>")
DATABASE = "dbaiinv" # os.getenv("ADX_DATABASE", "<your_database>")
AAD_APP_ID = "594" # os.getenv("AAD_APP_ID", "<your_app_id>")
AAD_APP_SECRET = "BR88Q~cA" # os.getenv("AAD_APP_SECRET", "<your_app_secret>")
AUTHORITY_ID = "186286" # os.getenv("AUTHORITY_ID", "<your_authority_id>")


app = FastAPI(
    title="ADX Blur Detection API",
    description="Returns image blur scores above a certain threshold from ADX.",
    version="1.0.0"
)

# Azure Kusto Client Initialization
kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(
    CLUSTER, AAD_APP_ID, AAD_APP_SECRET, AUTHORITY_ID
)
client = KustoClient(kcsb)

# Pydantic Model (you can extend this if needed)
class ImageBlurScore(BaseModel):
    image: str
    hybrid_confidence_score: float
    predicted_label: str


@app.get("/")
def root():
    return {"message": "Hey, All OK"}

@app.get("/blur-scores", response_model=List[ImageBlurScore])
def get_blur_scores(threshold: float = Query(60.0, ge=0.0, le=100.0, description="Threshold between 0-100")):
    try:
        query = f"""
        {TABLE_NAME}
        | where hybrid_confidence_score > {threshold}
        | order by hybrid_confidence_score desc
        """
        response = client.execute(DATABASE, query)
        df = pd.DataFrame([row.to_dict() for row in response.primary_results[0]])

        if df.empty:
            return []

        result = df[['image', 'hybrid_confidence_score', 'predicted_label']].to_dict(orient="records")
        # result = df_sorted.to_dict(orient="records") # return all column values
        
        return result # df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch results: {str(e)}")

# http://0.0.0.0:8000/blur-scores

# uvicorn fetch_data_from_adx_flaskapi:app --host 0.0.0.0 --port 8000 --reload
