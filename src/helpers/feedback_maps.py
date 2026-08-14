from enum import Enum


class FeedbackMap(tuple[int, int], Enum):
    Rating = (0, 3)

class HotRatingMap(int, Enum):
    Popular = FeedbackMap.Rating.value[1]
    Trending = FeedbackMap.Rating.value[1]

feedback_dict = {k: v.value for k, v in FeedbackMap._member_map_.items()}
hot_ratings_dict = {k: v.value for k, v in HotRatingMap._member_map_.items()}
