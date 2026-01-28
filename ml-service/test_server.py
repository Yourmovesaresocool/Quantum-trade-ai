from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "FastAPI is working inside the venv!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)