from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root_controller():
  return {"status": "healthy"}

if __name__ == "__main__":
  uvicorn.run("main:app", port=8000, reload=True)