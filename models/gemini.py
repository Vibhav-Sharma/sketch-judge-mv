from __future__ import annotations
import os, re, logging, time, random
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import google.generativeai as genai
from .base import AbstractModel, register_model


def _build_content(sample: Dict[str, Any]) -> List[Any]:
    query = sample["query"]
    img_tokens = re.findall(r"<(image_\d+)>", query)
    text_parts = re.split(r"<image_\d+>", query)

    content: List[Any] = []
    for idx, txt in enumerate(text_parts):
        if txt and txt.strip():
            content.append(txt)
        if idx < len(img_tokens):
            key = img_tokens[idx]
            img_obj = sample.get(key)
            if img_obj is None:
                logging.warning("Missing image for %s", key)
                continue
            if isinstance(img_obj, (str, Path)):
                img_obj = Image.open(img_obj)
            content.append(img_obj)

    return content


@register_model("gemini")
class GeminiModel(AbstractModel):
    def __init__(
        self,
        model_name: str = "gemini-1.5-pro",
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
        retry_attempts: int = 3,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_attempts = retry_attempts

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("API key for Gemini is required")

        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        )

    def generate_from_sample(self, sample: Dict[str, Any]) -> str:
        content = _build_content(sample)

        for attempt in range(1, 100):
            try:
                resp = self.model.generate_content(content)
                text = resp.text
                return text.strip() if text else ""
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
                    sleep_s = 65
                    logging.warning(f"Rate limit hit (attempt {attempt}): {e}. Sleep for {sleep_s}s.")
                else:
                    if attempt >= self.retry_attempts:
                        logging.error("All retries failed: %s", e)
                        break
                    sleep_s = min(30, 2 ** (attempt - 1) + random.random())
                    logging.warning(f"Attempt {attempt} failed: {e}. Retry in {sleep_s:.1f}s.")
                time.sleep(sleep_s)

        return ""
