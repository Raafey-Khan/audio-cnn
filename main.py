import modal
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import librosa
import io
from typing import List, Dict
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = modal.App("audio-cnn-inference")


fastapi_app = FastAPI()


fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class AudioInput(BaseModel):
    audio_data: str

class Prediction(BaseModel):
    class_name: str  # Changed to class_name to match frontend
    confidence: float

class LayerData(BaseModel):
    shape: List[int]
    values: List[List[float]]

class WaveformData(BaseModel):
    values: List[float]
    sample_rate: int
    duration: float

class ApiResponse(BaseModel):
    predictions: List[Prediction]
    visualization: Dict[str, LayerData]
    input_spectrogram: LayerData
    waveform: WaveformData


class AudioClassifier:
    def __init__(self):
       
        self.model = None
        logger.info("Model loading placeholder")

    def predict(self, audio: np.ndarray, sample_rate: int) -> dict:
        
        try:
            logger.info("Starting prediction")
         
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

            
            predictions = [
                {"class_name": "helicopter", "confidence": 0.1915},
                {"class_name": "rooster", "confidence": 0.1783},
                {"class_name": "crying_baby", "confidence": 0.1085},
            ]

          
            visualization = {
                "conv1": {
                    "shape": [64, 64],
                    "values": np.random.rand(64, 64).tolist()
                },
                "conv1.relu": {
                    "shape": [64, 64],
                    "values": np.random.rand(64, 64).tolist()
                }
            }

          
            input_spectrogram = {
                "shape": list(spectrogram.shape),
                "values": spectrogram.tolist()
            }

         
            waveform = {
                "values": audio[:1000].tolist(),  # First 1000 samples
                "sample_rate": sample_rate,
                "duration": len(audio) / sample_rate
            }

            logger.info("Prediction completed")
            return {
                "predictions": predictions,
                "visualization": visualization,
                "input_spectrogram": input_spectrogram,
                "waveform": waveform
            }
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise


classifier = AudioClassifier()


@fastapi_app.options("/")
async def options_handler(request: Request):
    logger.info("Handling OPTIONS request")
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "http://localhost:3000",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )


@app.function(
    image=modal.Image.debian_slim().pip_install(
        "fastapi==0.103.0",
        "pydantic==2.4.2",
        "librosa==0.10.1",
        "numpy==1.24.3"
    ),
    secrets=[],
)
@modal.asgi_app()
def inference():
    @fastapi_app.on_event("startup")
    async def startup_event():
        logger.info("Loading models on enter")
      
        logger.info("Model loaded on enter")

    @fastapi_app.post("/", response_model=ApiResponse)
    async def predict(input: AudioInput):
        logger.info("Received POST request")
        try:
           
            audio_bytes = base64.b64decode(input.audio_data)
            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

        
            result = classifier.predict(audio, sr)

          
            logger.info(f"First 10 values: {result['waveform']['values'][:10]}...")
            logger.info(f"Duration: {result['waveform']['duration']}")
            logger.info("Top predictions:")
            for pred in result["predictions"]:
                logger.info(f"  -{pred['class_name']} {pred['confidence']*100:.2f}%")

            return ApiResponse(
                predictions=[Prediction(class_name=p["class_name"], confidence=p["confidence"]) for p in result["predictions"]],
                visualization={k: LayerData(shape=v["shape"], values=v["values"]) for k, v in result["visualization"].items()},
                input_spectrogram=LayerData(shape=result["input_spectrogram"]["shape"], values=result["input_spectrogram"]["values"]),
                waveform=WaveformData(
                    values=result["waveform"]["values"],
                    sample_rate=result["waveform"]["sample_rate"],
                    duration=result["waveform"]["duration"]
                )
            )
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            raise

    return fastapi_app


@app.local_entrypoint()
def main():
    logger.info("Running local test")
    audio = np.zeros(44100)  
    result = classifier.predict(audio, 44100)
    logger.info(f"Test result: {result['predictions'][:1]}")
