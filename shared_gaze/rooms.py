import re
import secrets


ROOM_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def generate_room_code(length: int = 6) -> str:
    if length < 4:
        raise ValueError("Room codes must be at least 4 characters")
    raw = "".join(secrets.choice(ROOM_ALPHABET) for _ in range(length))
    if length == 6:
        return f"{raw[:3]}-{raw[3:]}"
    return raw


def normalize_room_code(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    if not 4 <= len(normalized) <= 16:
        raise ValueError("Room code must be 4 to 16 letters or digits")
    return normalized

