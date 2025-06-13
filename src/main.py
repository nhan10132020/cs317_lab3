from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import uvicorn
import pandas as pd
from datetime import datetime
import logging
import traceback
import time 
import observability

observability.config({
	"otel_endpoint_url": "http://localhost:4317", 
    "service_name": "wine_classifier", 
    "format": "%(asctime)s - %(levelname)s -  %(message)s" 
})

tracer = observability.trace.get_tracer(__name__)
meter = observability.metrics.get_meter(__name__)

request_counter = meter.create_counter(
    name="http_server_requests",
    unit="1",
    description="Total number of incoming HTTP requests"
)

error_counter = meter.create_counter(
    name="http_server_errors",
    unit="1",
    description="Total number of failed HTTP requests"
)

latency_histogram = meter.create_histogram(
    name="http_server_duration",
    unit="ms",
    description="The duration of HTTP requests"
)

cpu_timer = meter.create_histogram("model_inference_cpu_time", unit="s", description="CPU time for inference")

MODEL_PATH = "src/model/wine_classification.pkl"  
SCALER_PATH = "src/model/scaler.joblib"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {e}")

class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float = Field(..., alias="od280/od315_of_diluted_wines")
    proline: float

app = FastAPI(title="Wine Classifier API", version="1.0")

@app.post("/predict")
def predict_wine(input: WineInput):
    # calculate request latency
    start_time = time.time()
    start_cpu = time.perf_counter()
    request_counter.add(1, {"method": "POST", "endpoint": "/predict"})
    with tracer.start_as_current_span(f'Prediction At: {datetime.now().strftime("%H:%M:%S")}') as parent:
        try:
            with tracer.start_as_current_span('Input Validation'):
                logging.info("Received input for prediction")
                input_dict = input.model_dump(by_alias=True)  
            
                input_df = pd.DataFrame([input_dict])  

                input_scaled = scaler.transform(input_df)
                
                logging.info("Input data scaled successfully")
                prediction = model.predict(input_scaled)[0]
            with tracer.start_as_current_span(f'Model Prediction'):
                logging.info(f"Prediction made")
                probability = model.predict_proba(input_scaled).tolist()[0]
                logging.info(f"Prediction Done!")
                latency = (time.time() - start_time) * 1000  
                cpu_time_sec = time.perf_counter() - start_cpu
                cpu_timer.record(cpu_time_sec)
                latency_histogram.record(latency, {"method": "GET", "endpoint": "/api"})
                return {
                    "predicted_class": int(prediction),
                    "probability": probability
                }

        except Exception as e:
            error_counter.add(1, {"status_code": "500", "endpoint": "/predict"})
            with tracer.start_as_current_span('Error Handling'):
                logging.error(f"Error during prediction: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            
# error for example
@app.get("/home")
def error_example():
    request_counter.add(1, {"method": "GET", "endpoint": "/home"})
    with tracer.start_as_current_span('Error Example'):
        try:
            raise ValueError("This is a simulated error for testing purposes.")
        except ValueError as e:
            error_counter.add(1, {"status_code": "500", "endpoint": "/home"})
            logging.error(f"Simulated error occurred: {e}")
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

# Example delete method
@app.delete("/delete")
def delete_example():
    request_counter.add(1, {"method": "DELETE", "endpoint": "/delete"})
    with tracer.start_as_current_span('Delete Example'):
        try:
            # Simulate a delete operation
            logging.info("Delete operation successful")
            return {"message": "Delete operation successful"}
        except Exception as e:
            error_counter.add(1, {"status_code": "500", "endpoint": "/delete"})
            logging.error(f"Error during delete operation: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
# Example put method
@app.put("/update")
def update_example():
    request_counter.add(1, {"method": "PUT", "endpoint": "/update"})
    with tracer.start_as_current_span('Update Example'):
        try:
            # Simulate an update operation
            logging.info("Update operation successful")
            return {"message": "Update operation successful"}
        except Exception as e:
            error_counter.add(1, {"status_code": "500", "endpoint": "/update"})
            logging.error(f"Error during update operation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Example patch method
@app.patch("/modify")
def modify_example():
    request_counter.add(1, {"method": "PATCH", "endpoint": "/modify"})
    with tracer.start_as_current_span('Modify Example'):
        try:
            # Simulate a modify operation
            logging.info("Modify operation successful")
            return {"message": "Modify operation successful"}
        except Exception as e:
            error_counter.add(1, {"status_code": "500", "endpoint": "/modify"})
            logging.error(f"Error during modify operation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
