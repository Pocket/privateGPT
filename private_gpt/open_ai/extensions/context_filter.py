from pydantic import BaseModel, Field


class ContextFilter(BaseModel):
    user_id: str = Field(examples=["123"])
