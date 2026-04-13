from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    temperature: float = 0.2
    max_new_tokens: int = 512


class DownloadRequest(BaseModel):
    url: str
    filename: str


class BulkDownloadRequest(BaseModel):
    pdfs: list[DownloadRequest]
