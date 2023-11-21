from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from private_gpt.server.ingest.ingest_service import IngestedDoc, IngestService
from private_gpt.server.utils.auth import authenticated
from private_gpt.server.utils.user import User

ingest_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class IngestResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[IngestedDoc]


@ingest_router.post(
    "/ingest", tags=["Ingestion"], dependencies=[Depends(authenticated)]
)
def ingest(
    request: Request,
    file: Annotated[UploadFile, File],
    url: Annotated[str, Form()],
    item_id: Annotated[str, Form()],
    user: Annotated[User, Depends(authenticated)],
) -> IngestResponse:
    """Ingests and processes a file, storing its chunks to be used as context.

    The context obtained from files is later used in
    `/chat/completions`, `/completions`, and `/chunks` APIs.

    Most common document
    formats are supported, but you may be prompted to install an extra dependency to
    manage a specific file type.

    A file can generate different Documents (for example a PDF generates one Document
    per page). All Documents IDs are returned in the response, together with the
    extracted Metadata (which is later used to improve context retrieval). Those IDs
    can be used to filter the context used to create responses in
    `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    if not user.allowed_ingest:
        raise HTTPException(401, "Not authorized to ingest content")

    service = request.state.injector.get(IngestService)
    if file.filename is None:
        raise HTTPException(400, "No file name provided")
    service.delete_item(user_id=user.sub, item_id=item_id)
    ingested_documents = service.ingest(
        file.filename, file.file.read(), user_id=user.sub, url=url, item_id=item_id
    )
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.get(
    "/ingest/list", tags=["Ingestion"], dependencies=[Depends(authenticated)]
)
def list_ingested(
    request: Request,
    user: Annotated[User, Depends(authenticated)],
) -> IngestResponse:
    """Lists already ingested Documents including their Document ID and metadata.

    Those IDs can be used to filter the context used to create responses
    in `/chat/completions`, `/completions`, and `/chunks` APIs.
    """
    if not user.allowed_ingest:
        raise HTTPException(401, "Not authorized to list ingested content")

    service = request.state.injector.get(IngestService)
    ingested_documents = service.list_ingested_user(user_id=user.sub)
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)


@ingest_router.delete(
    "/ingest/{doc_id}", tags=["Ingestion"], dependencies=[Depends(authenticated)]
)
def delete_ingested(
    request: Request,
    doc_id: str,
    user: Annotated[User, Depends(authenticated)],
) -> None:
    """Delete the specified ingested Document.

    The `doc_id` can be obtained from the `GET /ingest/list` endpoint.
    The document will be effectively deleted from your storage context.
    """
    if not user.allowed_ingest:
        raise HTTPException(401, "Not authorized to delete ingested content")
    service = request.state.injector.get(IngestService)
    service.delete(doc_id)


@ingest_router.delete(
    "/ingest/item/{item_id}", tags=["Ingestion"], dependencies=[Depends(authenticated)]
)
def delete_ingested_item(
    request: Request,
    item_id: str,
    user: Annotated[User, Depends(authenticated)],
) -> None:
    """Delete all documents related to the user and itemId."""
    if not user.allowed_ingest:
        raise HTTPException(401, "Not authorized to delete ingested content")
    service = request.state.injector.get(IngestService)
    service.delete_item(user_id=user.sub, item_id=item_id)
