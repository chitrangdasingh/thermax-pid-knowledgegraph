"""Optional REST API. Run: uvicorn api:app --host 127.0.0.1 --port 8000"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from platform_core import DB, asset_context, bootstrap, connect, ingest_observations, namespaces, read_table, summarize_asset
import pandas as pd
from dexpi_graphrag import condensed_graph, graph_context, load_current_graph, local_answer

@asynccontextmanager
async def lifespan(app):
    bootstrap()
    yield


app=FastAPI(title="Thermax P&ID POC API",version="0.1.0",lifespan=lifespan)


def auth(x_api_key):
    secret=os.environ.get("PID_API_KEY")
    if not secret:raise HTTPException(503,"Set PID_API_KEY before enabling API access")
    import secrets
    if not x_api_key or not secrets.compare_digest(x_api_key,secret):raise HTTPException(401,"Invalid API key")


@app.get("/health")
def health():
    return {"status":"ok","database_exists":DB.exists()}


@app.get("/assets")
def assets(x_api_key: str | None=Header(default=None)):
    auth(x_api_key)
    return read_table("assets").to_dict("records")


@app.get("/assets/{asset_id}")
def asset(asset_id:str,x_api_key:str|None=Header(default=None)):
    auth(x_api_key)
    ctx=asset_context(asset_id)
    if ctx is None:raise HTTPException(404,"Asset not found")
    return ctx


@app.get("/graph")
def graph(x_api_key:str|None=Header(default=None)):
    auth(x_api_key)
    return {"nodes":read_table("assets").to_dict("records"),"relationships":read_table("relationships").to_dict("records")}


@app.get("/namespace")
def namespace(x_api_key:str|None=Header(default=None)):
    auth(x_api_key)
    return namespaces()


@app.get("/alerts")
def alerts(x_api_key:str|None=Header(default=None)):
    auth(x_api_key)
    return read_table("alerts").to_dict("records")


@app.get("/copilot/{asset_id}")
def copilot(asset_id:str,x_api_key:str|None=Header(default=None)):
    auth(x_api_key)
    return summarize_asset(asset_id)


class GraphQuestion(BaseModel):
    question:str
    include_context:bool=False


@app.post("/qa")
def graph_qa(request:GraphQuestion,x_api_key:str|None=Header(default=None)):
    auth(x_api_key)
    if not request.question.strip() or len(request.question)>1000:
        raise HTTPException(422,"question must contain 1-1000 characters")
    pid_graph=condensed_graph(load_current_graph(),include_lexical=True)
    response=local_answer(request.question,pid_graph)
    if request.include_context:
        response["context"]=graph_context(pid_graph,request.question)
    return response


class Observation(BaseModel):
    timestamp:str
    sensor_tag:str
    asset_id:str
    value:float
    unit:str
    quality_code:str


@app.post("/observations")
def observations(rows:list[Observation],x_api_key:str|None=Header(default=None)):
    auth(x_api_key)
    if len(rows)>1000:raise HTTPException(413,"Maximum 1000 rows per request")
    return ingest_observations(pd.DataFrame([r.model_dump() for r in rows]),source="REST")
