from gql import Client, gql
from gql.client import AsyncClientSession
from gql.transport.aiohttp import AIOHTTPTransport
from pydantic import BaseModel, Field, AliasPath, TypeAdapter, ValidationError
import logging
from config.constants import HARDCOVER_URL

logger = logging.getLogger(__name__)


class Cover(BaseModel):
    id: int = Field(validation_alias=AliasPath("default_cover_edition", "id"))
    isbn_13: str = Field(validation_alias=AliasPath("default_cover_edition", "isbn_13"))
    book_id: int = Field(alias="id")
    image_url: str = Field(validation_alias=AliasPath("default_cover_edition", "image", "url"))

cover_adapter = TypeAdapter(Cover)

def get_hardcover_client(hardcover_token: str):
    transport = AIOHTTPTransport(
        url=HARDCOVER_URL,
        headers={
            "Authorization": f"Bearer {hardcover_token}",
        },
    )
    client = Client(transport=transport)

    return client

async def get_popular_covers(session: AsyncClientSession, popular_count: int) -> list[Cover]:
    query = gql(
        """
        query PopularCovers($popular_count: Int) {
            books(
                order_by: [{users_count: desc}]
                limit: $popular_count
            ) {
                id
                title
                
                default_cover_edition {
                    id
                    isbn_13
                    image {
                        url
                    }
                    language {
                        language
                    }
                    users_count
                    users_read_count
                    score
                }
            }
        }
        """
    )
    query.variable_values = {"popular_count": popular_count}

    result = await session.execute(query)
    logger.info("Popular Covers result: %s", result)

    covers = []
    for book in result["books"]:
        try:
            covers.append(cover_adapter.validate_python(book))
        except ValidationError:
            logger.warning("Invalid cover: %s", book)
            logger.info("Skipping cover with id: %s", book["default_cover_edition"]["id"])

    return covers

def order_covers(covers: list[Cover], id_list: list[int]) -> list[Cover]:
    covers_map = {cid: ind for ind, cid in enumerate(id_list)}
    ordered_covers = sorted(covers, key=lambda x: covers_map[x.book_id])

    return ordered_covers

async def get_trending_covers(session: AsyncClientSession, trending_count: int) -> list[Cover]:
    ids_query = gql(
        """
        query TrendingIds($trending_count: Int) {
            books_trending(
                duration: month,
                limit: $trending_count
            ) {
                ids
            }
        }
        """
    )
    ids_query.variable_values = {"trending_count": trending_count}

    ids_result = await session.execute(ids_query)
    id_list = ids_result["books_trending"]["ids"]
    if len(id_list) == 0:
        logger.warning("No trending books found")
        return []

    cover_query = gql(
        """
        query TrendingCovers($id_list: [Int]) {
            books(
                where: {
                    id: {_in: $id_list}
                }
            ) {
                id
                title

                default_cover_edition {
                    id
                    isbn_13
                    image {
                        url
                    }
                    language {
                        language
                    }
                    users_count
                    users_read_count
                    score
                }
            }
        }
        """
    )
    cover_query.variable_values = {"id_list": id_list}

    covers_result = await session.execute(cover_query)
    logger.info("Trending Covers result: %s", covers_result)

    covers = []
    for book in covers_result["books"]:
        try:
            covers.append(cover_adapter.validate_python(book))
        except ValidationError:
            logger.warning("Invalid cover: %s", book)
            logger.info("Skipping cover with id: %s", book["default_cover_edition"]["id"])

    return order_covers(covers, id_list)
