from enum import Enum


class RuntimeMode(str, Enum):
    NONE = "none"
    REPLAY = "replay"
    LIVE = "live"


current_mode = RuntimeMode.NONE
