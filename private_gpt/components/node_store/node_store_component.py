import logging
from importlib import util

from injector import inject, singleton
from llama_index.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.storage.index_store.types import BaseIndexStore

from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class NodeStoreComponent:
    index_store: BaseIndexStore
    doc_store: BaseDocumentStore

    @inject
    def __init__(self, settings: Settings) -> None:
        match settings.indexstore.database:
            case "disk":
                try:
                    self.index_store = SimpleIndexStore.from_persist_dir(
                        persist_dir=str(local_data_path)
                    )
                except FileNotFoundError:
                    logger.debug("Local index store not found, creating a new one")
                    self.index_store = SimpleIndexStore()
            case "redis":
                if util.find_spec("redis") is None:
                    raise ImportError(
                        "'redis' is not installed."
                        "To use PrivateGPT with Redis, install the 'redis' extra."
                        "`poetry install --extras redis`"
                    )
                from llama_index.storage.index_store import RedisIndexStore

                try:
                    self.index_store = RedisIndexStore.from_host_and_port(
                        host=settings.indexstore.redis.host,
                        port=settings.indexstore.redis.port,
                        namespace=settings.indexstore.namespace,
                    )
                except ValueError as e:
                    logger.error(e)
                    raise e
            case "dynamodb":
                from llama_index.storage.index_store.dynamodb_index_store import (
                    DynamoDBIndexStore,
                )

                try:
                    self.index_store = DynamoDBIndexStore.from_table_name(
                        table_name=settings.indexstore.dynamodb.table_name,
                        namespace=settings.indexstore.namespace,
                    )
                except ValueError as e:
                    logger.error(e)
                    raise e

        match settings.documentstore.database:
            case "disk":
                try:
                    self.doc_store = SimpleDocumentStore.from_persist_dir(
                        persist_dir=str(local_data_path)
                    )
                except FileNotFoundError:
                    logger.debug("Local document store not found, creating a new one")
                    self.doc_store = SimpleDocumentStore()
            case "redis":
                if util.find_spec("redis") is None:
                    raise ImportError(
                        "'redis' is not installed."
                        "To use PrivateGPT with Redis, install the 'redis' extra."
                        "`poetry install --extras redis`"
                    )
                try:
                    from llama_index.storage.docstore import RedisDocumentStore

                    self.doc_store = RedisDocumentStore.from_host_and_port(
                        host=settings.documentstore.redis.host,
                        port=settings.documentstore.redis.port,
                        namespace=settings.documentstore.namespace,
                    )
                except ValueError as e:
                    logger.error(e)
                    raise e
            case "dynamodb":
                from llama_index.storage.docstore.dynamodb_docstore import (
                    DynamoDBDocumentStore,
                )

                try:
                    self.doc_store = DynamoDBDocumentStore.from_table_name(
                        table_name=settings.documentstore.dynamodb.table_name,
                        namespace=settings.documentstore.namespace,
                    )
                except ValueError as e:
                    logger.error(e)
                    raise e
