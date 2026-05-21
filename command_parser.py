import re
from dataclasses import dataclass
from typing import Optional


CN_DIGITS = {
    "零": 0,
    "一": 1,
    "幺": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

DIRECTIONS = ["东北", "东南", "西北", "西南", "东", "南", "西", "北"]
TASK_ALIASES = {
    "侦查": "侦查",
    "侦察": "侦查",
    "打击": "打击",
    "警卫": "警卫",
}


@dataclass
class CommandSlots:
    wake_word: Optional[str] = None
    object: Optional[str] = None
    direction: Optional[str] = None
    distance_m: Optional[int] = None
    task: Optional[str] = None

    @property
    def valid(self) -> bool:
        return all(
            [self.wake_word, self.object, self.direction, self.distance_m is not None, self.task]
        )

    def to_dict(self):
        return {
            "valid": self.valid,
            "wake_word": self.wake_word,
            "object": self.object,
            "direction": self.direction,
            "distance_m": self.distance_m,
            "task": self.task,
        }


class CommandParser:
    def parse(self, text: str, wake_word: str) -> dict:
        normalized = self._normalize_text(text)
        wake_word = self._normalize_text(wake_word) or "小艾同学"
        command_text = self.strip_wake_word(normalized, wake_word)
        slots = CommandSlots(
            wake_word=wake_word,
            object=self._find_object(command_text),
            direction=self._find_direction(command_text),
            distance_m=self._find_distance(command_text),
            task=self._find_task(command_text),
        )
        return slots.to_dict()

    def contains_wake_word(self, text: str, wake_word: str) -> bool:
        normalized = self._normalize_text(text)
        wake_word = self._normalize_text(wake_word)
        return bool(wake_word and wake_word in normalized)

    def contains_control_word(self, text: str, control_word: str) -> bool:
        normalized = self._normalize_text(text)
        control_word = self._normalize_text(control_word)
        return bool(control_word and control_word in normalized)

    def strip_control_word(self, text: str, control_word: str) -> str:
        normalized = self._normalize_text(text)
        control_word = self._normalize_text(control_word)
        if control_word and control_word in normalized:
            return normalized.split(control_word, 1)[0]
        return normalized

    def extract_command_text(self, text: str, wake_word: str) -> str:
        normalized = self._normalize_text(text)
        wake_word = self._normalize_text(wake_word)
        return self.strip_wake_word(normalized, wake_word)

    def strip_wake_word(self, text: str, wake_word: str) -> str:
        if wake_word and wake_word in text:
            return text.split(wake_word, 1)[1]
        return text

    def _normalize_text(self, text: str) -> str:
        text = text.strip()
        text = text.replace(" ", "")
        text = text.replace("，", "").replace("。", "").replace(",", "")
        text = text.replace("：", "").replace(":", "")
        text = text.replace("机器人", "号机器人") if re.search(r"\d机器人", text) else text
        return text

    def _find_object(self, text: str) -> Optional[str]:
        match = re.search(r"([0-9]{1,2})号?(?:机器|机)?人?", text)
        if match:
            return f"{int(match.group(1))}号机器人"

        match = re.search(r"([零一幺二两三四五六七八九十]{1,3})号?(?:机器|机)?人?", text)
        if match:
            number = self._cn_to_int(match.group(1))
            if number is not None:
                return f"{number}号机器人"
        return None

    def _find_direction(self, text: str) -> Optional[str]:
        for direction in DIRECTIONS:
            if direction in text:
                return direction
        return None

    def _find_distance(self, text: str) -> Optional[int]:
        match = re.search(r"([0-9]{1,4})\s*米", text)
        if match:
            return int(match.group(1))

        match = re.search(r"([零一幺二两三四五六七八九十百]{1,6})米", text)
        if match:
            return self._cn_to_int(match.group(1))
        return None

    def _find_task(self, text: str) -> Optional[str]:
        for alias, task in TASK_ALIASES.items():
            if alias in text:
                return task
        return None

    def _cn_to_int(self, value: str) -> Optional[int]:
        if not value:
            return None
        if value.isdigit():
            return int(value)
        if value in CN_DIGITS:
            return CN_DIGITS[value]
        if "百" in value:
            left, _, right = value.partition("百")
            hundreds = CN_DIGITS.get(left, 1 if left == "" else None)
            if hundreds is None:
                return None
            rest = self._cn_to_int(right) if right else 0
            return hundreds * 100 + (rest or 0)
        if "十" in value:
            left, _, right = value.partition("十")
            tens = CN_DIGITS.get(left, 1 if left == "" else None)
            ones = CN_DIGITS.get(right, 0 if right == "" else None)
            if tens is None or ones is None:
                return None
            return tens * 10 + ones
        total = 0
        for char in value:
            digit = CN_DIGITS.get(char)
            if digit is None:
                return None
            total = total * 10 + digit
        return total
