from enum import Enum


class FeedbackMap(tuple[int, int], Enum):
    Rating = (0, 3)

class HotRatingMap(int, Enum):
    Popular = FeedbackMap.Rating.value[1]
    Trending = FeedbackMap.Rating.value[1]
