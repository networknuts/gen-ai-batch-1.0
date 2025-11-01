from fastapi import FastAPI, Query, Path
from rag_async.tasks.connection import queue          # <-- absolute import
from rag_async.tasks.worker import process_query      # <-- absolute import

app = FastAPI()

@app.get("/")
def root():
    return {"status": "Server is up and running"}

@app.post("/chat")
def chat(query: str = Query(..., description="Chat Message")):
    job = queue.enqueue(process_query, query)         # pass the callable
    return {"status": "queued", "job_id": job.id}

@app.get("/result/{job_id}")
def get_result(
    job_id: str = Path(..., description="Job ID")
):
    job = queue.fetch_job(job_id=job_id)
    result = job.return_value()

    return {"result": result}
