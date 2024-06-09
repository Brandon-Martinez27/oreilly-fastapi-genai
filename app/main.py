import uvicorn
from fastapi import FastAPI, Query, status
from fastapi.responses import StreamingResponse
from models import generate_text, load_text_model, load_audio_model, generate_audio
from schemas import VoicePresets
from utils import audio_array_to_buffer

app = FastAPI()


@app.get("/generate/text")
def serve_language_model_controller(prompt=Query):
    pipe = load_text_model()
    output = generate_text(pipe, prompt)
    return output


@app.get(
    "/generate/audio",
    responses={status.HTTP_200_OK: {"content": {"audio/wav": {}}}},
    response_class=StreamingResponse,
) 
def serve_text_to_audio_model_controller(
    prompt=Query(...),
    preset: VoicePresets = Query(default="v2/en_speaker_1"),
):
    processor, model = load_audio_model()
    output, sample_rate = generate_audio(processor, model, prompt, preset)
    return StreamingResponse(audio_array_to_buffer(output, sample_rate), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
