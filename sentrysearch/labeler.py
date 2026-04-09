"""Gemini-powered shot labeling for batch retrieval metadata generation."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .chunker import SUPPORTED_VIDEO_EXTENSIONS, scan_directory
from .gemini_embedder import (
    DEFAULT_RPM,
    GeminiAPIKeyError,
    _RateLimiter,
    _retry,
)

load_dotenv()

DEFAULT_LABEL_MODEL = "gemini-3.1-flash-lite-preview"

LABEL_PROMPT = """你是一个视频镜头标注器。你的任务是为“单个短视频镜头”生成严格 JSON 标签，用于后续检索、筛选和重新组合。

只分析当前镜头中实际可见、或可以从画面中明确判断的信息。
不要根据账号、上下文剧情、粉丝知识、常识或猜测补充内容。
如果不确定，请使用低置信度，或填写 null / [] / "未知"。
禁止输出 Markdown。
禁止输出解释。
禁止输出代码块。
只返回一个 JSON 对象。
返回结果必须符合下面的字段定义，不能新增字段，不能缺少字段。

字段规则：
- 所有键名必须完全一致。
- 所有数组内元素必须是简短短语，不要写长句。
- `tags`、`search_phrases`、`appearance`、`expression`、`pose_action`、`setting`、`objects`、`wardrobe`、`text_on_screen`、`graphic_elements`、`lighting_color`、`composition`、`camera_motion`、`transition_feel`、`aesthetic`、`best_use_cases`、`continuity_cues` 都必须去重。
- `summary` 用中文，1 到 2 句，客观描述。
- `search_phrases` 用中文，5 到 10 条，像真实用户会输入的搜索语句。
- `tags` 用中文，8 到 16 个，尽量短，便于检索。
- 如果没有识别出人物，`people` 返回空数组。
- 如果能识别人物但不确定姓名，`name` 填 null。
- `role` 只能是："主角"、"背景人物"、"群体"
- `pace` 只能是："慢"、"中"、"快"、"未知"
- `music_energy` 只能是："低"、"中"、"高"、"未知"
- `hook_strength` 只能是："低"、"中"、"高"、"未知"
- `confidence` 和 `people[*].confidence` 必须是 0 到 1 之间的小数。
- 如果画面里没有文字、图形、水印、字幕，对应字段返回空数组。
- 不要写“帅”“好看”“很美”这类空泛评价，除非它体现为明确可检索的镜头风格标签。
- 不要编造剧情。
- 不要引用镜头外的信息。

输出前自检：
1. 是否是合法 JSON
2. 是否只有一个 JSON 对象
3. 是否没有缺失字段
4. 是否没有多余字段
5. 是否没有 Markdown 或解释文字
6. 是否所有枚举值都在允许范围内
7. 是否所有未知信息都使用了 null、[] 或 "未知"
"""

LABEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "search_phrases",
        "tags",
        "people",
        "visual_elements",
        "editing_style",
        "recombination_notes",
        "safety_flags",
        "confidence",
    ],
    "properties": {
        "summary": {"type": "string"},
        "search_phrases": {
            "type": "array",
            "minItems": 5,
            "maxItems": 10,
            "uniqueItems": True,
            "items": {"type": "string"},
        },
        "tags": {
            "type": "array",
            "minItems": 8,
            "maxItems": 16,
            "uniqueItems": True,
            "items": {"type": "string"},
        },
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "role", "appearance", "expression", "pose_action", "confidence"],
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "role": {"type": "string", "enum": ["主角", "背景人物", "群体"]},
                    "appearance": {
                        "type": "array", "uniqueItems": True, "items": {"type": "string"},
                    },
                    "expression": {
                        "type": "array", "uniqueItems": True, "items": {"type": "string"},
                    },
                    "pose_action": {
                        "type": "array", "uniqueItems": True, "items": {"type": "string"},
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "visual_elements": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "setting",
                "objects",
                "wardrobe",
                "text_on_screen",
                "graphic_elements",
                "lighting_color",
                "composition",
                "camera_motion",
            ],
            "properties": {
                "setting": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
                "objects": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
                "wardrobe": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
                "text_on_screen": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
                "graphic_elements": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
                "lighting_color": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
                "composition": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
                "camera_motion": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
            },
        },
        "editing_style": {
            "type": "object",
            "additionalProperties": False,
            "required": ["pace", "transition_feel", "aesthetic", "music_energy"],
            "properties": {
                "pace": {"type": "string", "enum": ["慢", "中", "快", "未知"]},
                "transition_feel": {
                    "type": "array", "uniqueItems": True, "items": {"type": "string"},
                },
                "aesthetic": {
                    "type": "array", "uniqueItems": True, "items": {"type": "string"},
                },
                "music_energy": {"type": "string", "enum": ["低", "中", "高", "未知"]},
            },
        },
        "recombination_notes": {
            "type": "object",
            "additionalProperties": False,
            "required": ["best_use_cases", "continuity_cues", "hook_strength"],
            "properties": {
                "best_use_cases": {
                    "type": "array", "uniqueItems": True, "items": {"type": "string"},
                },
                "continuity_cues": {
                    "type": "array", "uniqueItems": True, "items": {"type": "string"},
                },
                "hook_strength": {"type": "string", "enum": ["低", "中", "高", "未知"]},
            },
        },
        "safety_flags": {
            "type": "object",
            "additionalProperties": False,
            "required": ["violence", "sexualized_content", "adult_theme"],
            "properties": {
                "violence": {"type": "boolean"},
                "sexualized_content": {"type": "boolean"},
                "adult_theme": {"type": "boolean"},
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

_ROLE_VALUES = {"主角", "背景人物", "群体"}
_PACE_VALUES = {"慢", "中", "快", "未知"}
_ENERGY_VALUES = {"低", "中", "高", "未知"}
_HOOK_VALUES = {"低", "中", "高", "未知"}


def _unique_strings(values) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values or []:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _clamp_confidence(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _enum_or_default(value, allowed: set[str], default: str) -> str:
    text = str(value).strip() if value is not None else default
    return text if text in allowed else default


def _output_path_for(video_path: str, output_dir: str | None = None) -> str:
    video = Path(video_path)
    directory = Path(output_dir).expanduser() if output_dir else video.parent
    return str(directory / f"{video.stem}.label.json")


def _normalize_label(data: dict) -> dict:
    people = []
    for person in data.get("people", []):
        people.append({
            "name": person.get("name"),
            "role": _enum_or_default(person.get("role"), _ROLE_VALUES, "背景人物"),
            "appearance": _unique_strings(person.get("appearance")),
            "expression": _unique_strings(person.get("expression")),
            "pose_action": _unique_strings(person.get("pose_action")),
            "confidence": _clamp_confidence(person.get("confidence")),
        })

    visual = data.get("visual_elements", {})
    editing = data.get("editing_style", {})
    recombination = data.get("recombination_notes", {})
    safety = data.get("safety_flags", {})

    return {
        "summary": str(data.get("summary", "")).strip(),
        "search_phrases": _unique_strings(data.get("search_phrases")),
        "tags": _unique_strings(data.get("tags")),
        "people": people,
        "visual_elements": {
            "setting": _unique_strings(visual.get("setting")),
            "objects": _unique_strings(visual.get("objects")),
            "wardrobe": _unique_strings(visual.get("wardrobe")),
            "text_on_screen": _unique_strings(visual.get("text_on_screen")),
            "graphic_elements": _unique_strings(visual.get("graphic_elements")),
            "lighting_color": _unique_strings(visual.get("lighting_color")),
            "composition": _unique_strings(visual.get("composition")),
            "camera_motion": _unique_strings(visual.get("camera_motion")),
        },
        "editing_style": {
            "pace": _enum_or_default(editing.get("pace"), _PACE_VALUES, "未知"),
            "transition_feel": _unique_strings(editing.get("transition_feel")),
            "aesthetic": _unique_strings(editing.get("aesthetic")),
            "music_energy": _enum_or_default(editing.get("music_energy"), _ENERGY_VALUES, "未知"),
        },
        "recombination_notes": {
            "best_use_cases": _unique_strings(recombination.get("best_use_cases")),
            "continuity_cues": _unique_strings(recombination.get("continuity_cues")),
            "hook_strength": _enum_or_default(recombination.get("hook_strength"), _HOOK_VALUES, "未知"),
        },
        "safety_flags": {
            "violence": bool(safety.get("violence", False)),
            "sexualized_content": bool(safety.get("sexualized_content", False)),
            "adult_theme": bool(safety.get("adult_theme", False)),
        },
        "confidence": _clamp_confidence(data.get("confidence")),
    }


class GeminiShotLabeler:
    """Generate strict JSON labels for individual shot clips."""

    def __init__(self, model: str = DEFAULT_LABEL_MODEL, max_per_minute: int = DEFAULT_RPM):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise GeminiAPIKeyError(
                "GEMINI_API_KEY is not set.\n\n"
                "Run: sentrysearch init"
            )
        self._client = genai.Client(api_key=api_key)
        self._limiter = _RateLimiter(max_per_minute=max_per_minute)
        self._model = model

    def label_video(self, video_path: str, verbose: bool = False) -> dict:
        from google.genai import types

        with open(video_path, "rb") as f:
            video_bytes = f.read()

        video_part = types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
        prompt_part = types.Part.from_text(text="请分析这个视频镜头，并严格按要求返回 JSON。")

        self._limiter.wait()
        response = _retry(
            lambda: self._client.models.generate_content(
                model=self._model,
                contents=types.Content(role="user", parts=[prompt_part, video_part]),
                config=types.GenerateContentConfig(
                    system_instruction=LABEL_PROMPT,
                    response_mime_type="application/json",
                    response_json_schema=LABEL_SCHEMA,
                    temperature=0.1,
                    top_p=0.8,
                    max_output_tokens=2048,
                ),
            )
        )

        if getattr(response, "parsed", None) is not None:
            data = response.parsed
        else:
            text = getattr(response, "text", "") or ""
            data = json.loads(text)

        normalized = _normalize_label(data)

        if verbose:
            print(
                f"[verbose] labeled {os.path.basename(video_path)} "
                f"with {len(normalized['tags'])} tags and "
                f"{len(normalized['search_phrases'])} search phrases",
            )

        return normalized


def label_videos(
    path: str,
    *,
    output_dir: str | None = None,
    model: str = DEFAULT_LABEL_MODEL,
    overwrite: bool = False,
    verbose: bool = False,
) -> dict:
    """Label one video or a directory of videos and save adjacent JSON files."""
    input_path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(input_path):
        videos = [input_path]
    else:
        videos = [os.path.abspath(video) for video in scan_directory(input_path)]

    if not videos:
        return {"processed": 0, "skipped": 0, "items": []}

    labeler = GeminiShotLabeler(model=model)
    items = []
    processed = 0
    skipped = 0

    for video_path in videos:
        label_path = _output_path_for(video_path, output_dir=output_dir)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        if not overwrite and os.path.exists(label_path):
            skipped += 1
            items.append({
                "video": video_path,
                "label_path": label_path,
                "status": "skipped",
            })
            continue

        label = labeler.label_video(video_path, verbose=verbose)
        with open(label_path, "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)
            f.write("\n")

        processed += 1
        items.append({
            "video": video_path,
            "label_path": label_path,
            "status": "labeled",
        })

    return {"processed": processed, "skipped": skipped, "items": items}

