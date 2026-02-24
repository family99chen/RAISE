from typing import Optional, Union

from .api import ApiLLM
from .local import LocalLLM


def create_llm(
    url: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Union[ApiLLM, LocalLLM]:
    if api_key and model_name:
        return ApiLLM(url=url, api_key=api_key, model_name=model_name, timeout=timeout)
    if api_key and not model_name:
        raise ValueError("model_name is required when api_key is provided.")
    return LocalLLM(url=url, model_name=model_name, timeout=timeout)


def main() -> None:
    url = "http://localhost:9001/v1"
    query = "你好，给我介绍一下openai。"
    llm = create_llm(url=url, timeout=30)
    reply = llm.generate(query)
    print(reply)


def main_api() -> None:
    url = "https://api.zhizengzeng.com/v1"
    model_name = "gpt-4o-mini"
    api_key = input("Enter API key: ").strip()
    query = "你好，给我介绍一下openai。"
    llm = create_llm(url=url, api_key=api_key, model_name=model_name, timeout=30)
    reply = llm.generate(query)
    print(reply)


if __name__ == "__main__":
    main()
