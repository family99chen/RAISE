import asyncio
import base64
from dataclasses import dataclass
import json
import os
import mimetypes
from typing import Any, Dict, List, Optional, Tuple
import yaml
from urllib import request
from urllib.error import HTTPError, URLError


@dataclass
class ApiLLM:
    url: str
    api_key: str
    model_name: Optional[str] = None
    timeout: Optional[float] = None

    def _load_defaults(self) -> Tuple[float, int, Optional[float]]:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        config_path = os.path.join(base_dir, "llmconfig.yaml")
        if not os.path.isfile(config_path):
            return 0.7, 256, self.timeout
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        generator_cfg = data.get("generator", {})
        return (
            generator_cfg.get("temperature", 0.7),
            generator_cfg.get("max_tokens", 256),
            generator_cfg.get("timeout", self.timeout),
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _image_to_data_url(self, path: str) -> str:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or "application/octet-stream"
        with open(path, "rb") as handle:
            data = base64.b64encode(handle.read()).decode("utf-8")
        return f"data:{mime};base64,{data}"

    def _build_image_content(self, images: List[str]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for item in images:
            if not item:
                continue
            if item.startswith("http://") or item.startswith("https://"):
                content.append({"type": "image_url", "image_url": {"url": item}})
                continue
            if os.path.isfile(item):
                content.append(
                    {"type": "image_url", "image_url": {"url": self._image_to_data_url(item)}}
                )
        return content

    def _request_json(self, method: str, url: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        last_exc: Exception | None = None
        for _ in range(3):
            req = request.Request(url, data=data, method=method, headers=self._headers())
            try:
                with request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"API request failed after 3 attempts: {last_exc}") from last_exc

    def _resolve_model_name(self) -> str:
        if self.model_name:
            return self.model_name
        models_url = self.url.rstrip("/") + "/models"
        data = self._request_json("GET", models_url)
        models = data.get("data", [])
        if not models:
            raise RuntimeError("No models found from /v1/models.")
        model_id = models[0].get("id")
        if not model_id:
            raise RuntimeError("Model id missing in /v1/models response.")
        self.model_name = model_id
        return model_id

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        default_temperature, default_max_tokens, default_timeout = self._load_defaults()
        if temperature is None:
            temperature = default_temperature
        if max_tokens is None:
            max_tokens = default_max_tokens
        if default_timeout is not None:
            self.timeout = default_timeout
        try:
            model = self._resolve_model_name()
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            url = self.url.rstrip("/") + "/chat/completions"
            data = self._request_json("POST", url, payload)
            return data["choices"][0]["message"]["content"]
        except Exception:
            return ""

    def generate_multimodal(
        self,
        prompt: str,
        images: List[str],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        default_temperature, default_max_tokens, default_timeout = self._load_defaults()
        if temperature is None:
            temperature = default_temperature
        if max_tokens is None:
            max_tokens = default_max_tokens
        if default_timeout is not None:
            self.timeout = default_timeout
        try:
            model = self._resolve_model_name()
            messages = []
            if system:
                messages.append({"role": "system", "content": system})

            user_content: List[Dict[str, Any]] = []
            if prompt:
                user_content.append({"type": "text", "text": prompt})
            user_content.extend(self._build_image_content(images))
            if not user_content:
                return ""

            messages.append({"role": "user", "content": user_content})
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            url = self.url.rstrip("/") + "/chat/completions"
            data = self._request_json("POST", url, payload)
            return data["choices"][0]["message"]["content"]
        except Exception:
            return ""

    async def generate_multimodal_async(
        self,
        prompt: str,
        images: List[str],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return await asyncio.to_thread(
            self.generate_multimodal,
            prompt,
            images,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def generate_async(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        return await asyncio.to_thread(
            self.generate,
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
