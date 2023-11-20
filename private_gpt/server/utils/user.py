from pydantic import BaseModel


class User(BaseModel):
    sub: str
    allowed_ingest: bool
