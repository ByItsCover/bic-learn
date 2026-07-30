from helpers.hardcover import CoverRecord
from config.constants import HOT_FEEDBACK_TYPE


class FeedbackMap(tuple[int, int], Enum):
    Rating = (0, 3)


def get_hot_covers_map(
        popular_covers: list[CoverRecord], trending_covers: list[CoverRecord],
        feedback_type: str = HOT_FEEDBACK_TYPE
    ) -> dict[int, tuple[CoverRecord, float]]:
    popular_score = FeedbackMap[feedback_type].value[1]
    trending_score = FeedbackMap[feedback_type].value[1]

    popular_covers_map = {cover.id: (cover, popular_score) for cover in popular_covers}
    trending_covers_map = {cover.id: (cover, trending_score) for cover in trending_covers}
    return popular_covers_map | trending_covers_map
