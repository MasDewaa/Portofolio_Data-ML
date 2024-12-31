# This Python code snippet is setting up a FastAPI application for serving an API related to a
# Stanford Dogs Model. Here's a breakdown of what each part of the code is doing:
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import cnn

app = FastAPI(
    title="Stanford Dogs Model API",
    description="Stanford Dogs Model FastAPI",
)

# Konfigurasi CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sertakan router dari cnn.py
app.include_router(cnn.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
