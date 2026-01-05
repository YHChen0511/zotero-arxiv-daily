from openai import OpenAI
from loguru import logger
from time import sleep

GLOBAL_LLM = None


class LLM:
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        lang: str = "English",
    ):
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
        else:
            # For now, we strictly require API Key as we removed local LLM
            # Could raise an error or just let it fail later if used
            logger.warning("No API Key provided. LLM will not work.")
            self.llm = None

        self.model = model
        self.lang = lang

    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.chat.completions.create(
                        messages=messages, temperature=0, model=self.model
                    )
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    sleep(3)
            return response.choices[0].message.content
        else:
            raise ValueError(
                "Local LLM is no longer supported. Please provide an OpenAI-compatible API Key."
            )


def set_global_llm(
    api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"
):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)


def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.error("No global LLM found. Please call `set_global_llm` first.")
        raise ValueError("Global LLM not initialized")
    return GLOBAL_LLM
