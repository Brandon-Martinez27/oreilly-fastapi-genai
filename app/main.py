from io import BytesIO

import uvicorn
from fastapi import FastAPI, File, Query, Response, UploadFile, status
from fastapi.responses import RedirectResponse, StreamingResponse
from models import (
    generate_3d_geometry,
    generate_audio,
    generate_image,
    generate_text,
    generate_video,
    load_3d_model,
    load_audio_model,
    load_image_model,
    load_text_model,
    load_video_model,
)
from PIL import Image
from schemas import VoicePresets
from utils import (
    audio_array_to_buffer,
    export_to_video_buffer,
    img_to_bytes,
    mesh_to_obj_buffer,
)

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
    return StreamingResponse(
        audio_array_to_buffer(output, sample_rate), media_type="audio/wav"
    )


@app.get(
    "/generate/image",
    responses={status.HTTP_200_OK: {"content": {"image/png": {}}}},
    response_class=Response,
)
def serve_text_to_image_model_controller(prompt=Query(...)):
    pipe = load_image_model()
    output = generate_image(pipe, prompt)
    return Response(content=img_to_bytes(output), media_type="image/png")


@app.get("/", include_in_schema=False)
def docs_redirect_controller():
    """You can access the /docs page during local development
    whenever you visit the base url / of the service. In production,
    keep in mind that you will need to disable this redirection and
    the /docs routes for enhanced security. The root handler for base
    url /, can return the service version in production.
    """
    return RedirectResponse(url="/docs", status_code=status.HTTP_303_SEE_OTHER)


@app.post(
    "/generate/video",
    responses={status.HTTP_200_OK: {"content": {"video/mp4": {}}}},
    response_class=StreamingResponse,
)
async def serve_image_to_video_model_controller(
    image: UploadFile, num_frames: int = Query(default=25)
):
    image = Image.frombytes(
        data=BytesIO(await image.read()),
        mode="RGB",
        size=(image.size, image.size),
    )
    model = load_video_model()
    frames = generate_video(model, image, num_frames)
    return StreamingResponse(export_to_video_buffer(frames), media_type="video/mp4")


@app.get(
    "/generate/3d",
    responses={status.HTTP_200_OK: {"content": {"model/obj": {}}}},
    response_class=StreamingResponse,
)
def serve_text_to_3d_model_controller(
    prompt: str = Query(...), num_inference_steps: int = Query(default=25)
):
    model = load_3d_model()
    mesh = generate_3d_geometry(model, prompt, num_inference_steps)
    response = StreamingResponse(mesh_to_obj_buffer(mesh), media_type="model/obj")
    response.headers["Content-Disposition"] = f"attachment; filename={prompt}.obj"
    return response


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
