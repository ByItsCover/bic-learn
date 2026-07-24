import asyncio
import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError
import json
import logging
from helpers.hardcover import Cover

logger = logging.getLogger(__name__)


async def embed_covers(covers: list[Cover], lambda_client: BaseClient, function_name: str, get_log = False):
    await asyncio.to_thread(embed_covers_sync, covers, lambda_client, function_name, get_log)

def embed_covers_sync(covers: list[Cover], lambda_client: BaseClient, function_name: str, get_log = False):
    records = []
    for cover in covers:
        record = {
            "messageId": f"{cover.id}-{cover.isbn_13}",
            "body": cover.image_url,
            "messageAttributes": {
                "cover_id": {
                    "dataType": "Number",
                    "stringValue": cover.id,
                },
                "book_id": {
                    "dataType": "Number",
                    "stringValue": cover.book_id,
                },
                "isbn_13": {
                    "dataType": "String",
                    "stringValue": cover.isbn_13,
                },
            }
        }
        records.append(record)

    payload = {"Records": records}
    try:
        response = lambda_client.invoke(
            FunctionName=function_name,
            Payload=json.dumps(payload),
            LogType="Tail" if get_log else "None",
        )
        logger.info("Invoked function %s.", function_name)
    except ClientError:
        logger.exception("Couldn't invoke function %s.", function_name)
        raise
    return response

def get_lambda_client(aws_region: str):
    return boto3.client("lambda", region_name=aws_region)
