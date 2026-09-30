import json
import logging
import os
import re
from typing import Any, Dict, Optional, Union

import app.config as config

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM Client supporting both local Ollama and OpenAI providers,
    with structured schema generation, provider failover, and graceful offline fallback.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        # Resolve provider
        self.provider = (
            provider or os.getenv("LLM_PROVIDER") or config.LLM_PROVIDER or "ollama"
        ).lower()

        # If OPENAI_API_KEY is present and provider wasn't explicitly set to ollama, allow openai
        if (os.getenv("OPENAI_API_KEY") or config.OPENAI_API_KEY) and not provider:
            if os.getenv("LLM_PROVIDER") == "openai":
                self.provider = "openai"

        # Resolve model
        if model:
            self.model = model
        elif self.provider == "openai":
            self.model = config.OPENAI_MODEL
        else:
            self.model = config.OLLAMA_MODEL

        self.base_url = base_url or (
            config.OLLAMA_BASE_URL if self.provider == "ollama" else None
        )

        # Lazy initialized clients
        self._openai_client = None

    def _get_openai_client(self):
        if self._openai_client is None:
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY") or config.OPENAI_API_KEY
                self._openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize OpenAI client: {e}. "
                    "Ensure 'openai' package is installed and OPENAI_API_KEY is set."
                ) from e
        return self._openai_client

    # ========================================================
    # FALLBACK EXTRACTION LOGIC
    # ========================================================

    def _fallback_generate(self, input_text: str) -> str:
        lines = [line.strip() for line in input_text.splitlines() if line.strip() and not line.startswith("=")]
        summary = " ".join(lines[:3]) if lines else "Analysis based on grounded research context."
        return f"Based on the provided research context: {summary}"

    def _fallback_structured(self, schema: Any, input_text: str) -> str:
        """Construct a schema-valid response when external LLM endpoints are unavailable."""
        if hasattr(schema, "model_validate"):
            try:
                instance = schema()
                return instance.model_dump_json()
            except Exception:
                pass

        if hasattr(schema, "model_json_schema"):
            schema_dict = schema.model_json_schema()
            props = schema_dict.get("properties", {})
            dummy = {}
            for k, prop in props.items():
                ptype = prop.get("type")
                if ptype == "string":
                    dummy[k] = ""
                elif ptype == "array":
                    dummy[k] = []
                elif ptype == "object":
                    dummy[k] = {}
                else:
                    dummy[k] = None
            return json.dumps(dummy)

        return "{}"

    # ========================================================
    # NORMAL GENERATION
    # ========================================================

    def generate(
        self,
        instructions: str,
        input_text: str,
    ) -> str:
        try:
            if self.provider == "openai":
                return self._generate_openai(instructions, input_text)
            return self._generate_ollama(instructions, input_text)
        except Exception as primary_err:
            # Attempt failover to OpenAI if available
            api_key = os.getenv("OPENAI_API_KEY") or config.OPENAI_API_KEY
            if self.provider != "openai" and api_key:
                try:
                    return self._generate_openai(instructions, input_text)
                except Exception:
                    pass

            # If offline / unreachable, provide graceful fallback
            if os.getenv("STRICT_LLM") == "1":
                raise primary_err
            logger.warning(f"LLM generation offline fallback active: {primary_err}")
            return self._fallback_generate(input_text)

    def _generate_ollama(
        self,
        instructions: str,
        input_text: str,
    ) -> str:
        from ollama import chat

        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_text},
            ],
            options={"temperature": 0.1},
        )
        return response.message.content

    def _generate_openai(
        self,
        instructions: str,
        input_text: str,
    ) -> str:
        client = self._get_openai_client()
        model_name = config.OPENAI_MODEL if "qwen" in self.model else self.model
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_text},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""

    # ========================================================
    # STRUCTURED GENERATION
    # ========================================================

    def generate_structured(
        self,
        instructions: str,
        input_text: str,
        schema: Any,
    ) -> str:
        try:
            if self.provider == "openai":
                return self._generate_structured_openai(instructions, input_text, schema)
            return self._generate_structured_ollama(instructions, input_text, schema)
        except Exception as primary_err:
            # Attempt failover to OpenAI if available
            api_key = os.getenv("OPENAI_API_KEY") or config.OPENAI_API_KEY
            if self.provider != "openai" and api_key:
                try:
                    return self._generate_structured_openai(instructions, input_text, schema)
                except Exception:
                    pass

            # If offline / unreachable, return schema-valid fallback
            if os.getenv("STRICT_LLM") == "1":
                raise primary_err
            logger.warning(f"LLM structured generation offline fallback active: {primary_err}")
            return self._fallback_structured(schema, input_text)

    def _generate_structured_ollama(
        self,
        instructions: str,
        input_text: str,
        schema: Any,
    ) -> str:
        if hasattr(schema, "model_json_schema"):
            output_format = schema.model_json_schema()
        elif isinstance(schema, dict):
            output_format = schema
        else:
            output_format = {"type": "object"}

        from ollama import chat

        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_text},
            ],
            format=output_format,
            options={"temperature": 0.1},
        )
        return response.message.content

    def _generate_structured_openai(
        self,
        instructions: str,
        input_text: str,
        schema: Any,
    ) -> str:
        client = self._get_openai_client()
        model_name = config.OPENAI_MODEL if "qwen" in self.model else self.model
        if hasattr(schema, "model_json_schema"):
            completion = client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": input_text},
                ],
                response_format=schema,
                temperature=0.1,
            )
            parsed = completion.choices[0].message.parsed
            if parsed:
                return parsed.model_dump_json()
            return completion.choices[0].message.content or "{}"
        else:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": input_text},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return response.choices[0].message.content or "{}"
