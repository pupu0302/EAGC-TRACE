#!/usr/bin/env python3
"""
LLM Client abstraction layer
Supports multiple backends: OpenAI / Anthropic / Mock
"""

import os
import re
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Unified LLM call interface"""

    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}
        self.total_calls = 0
        self.total_cost = 0.0

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: str = "json_object",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Return format:
        {
            "content": str,           # generated content (JSON string or plain text)
            "parsed_json": dict,      # auto-parsed when response_format="json_object"
            "usage": {
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int
            },
            "model": str,
            "latency_ms": float,
            "cost_usd": float
        }
        """
        pass


class MockLLMClient(LLMClient):
    """Mock LLM (used for testing and fast iteration)"""

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
        logger.info(f"[MockLLM] Generating response for model: {self.model_name}")

        # Simulate latency
        time.sleep(0.1)

        # Extract information from user_prompt (simplified)
        mock_response = self._generate_mock_graph(user_prompt)

        return {
            "content": json.dumps(mock_response, indent=2),
            "parsed_json": mock_response,
            "usage": {
                "prompt_tokens": len(user_prompt) // 4,
                "completion_tokens": len(json.dumps(mock_response)) // 4,
                "total_tokens": (len(user_prompt) + len(json.dumps(mock_response))) // 4
            },
            "model": self.model_name,
            "latency_ms": 100,
            "cost_usd": 0.0
        }

    def _generate_mock_graph(self, prompt: str) -> Dict:
        """Intelligently generate a mock graph from the prompt (reuses the original logic)"""
        # Extract the evidence pool (simplified parsing)
        evidence_ids = []
        for line in prompt.split('\n'):
            if line.strip().startswith('[') and ']:' in line:
                evid_id = line.split(']')[0].strip('[')
                evidence_ids.append(evid_id)

        # Simple classification
        table_cells = [eid for eid in evidence_ids if ':cell:' in eid]
        definitions = [eid for eid in evidence_ids if ':def:' in eid]
        sentences = [eid for eid in evidence_ids if ':sent:' in eid]

        # Build the minimal graph
        nodes = [
            {"node_id": "n_metric", "type": "Metric", "text": "GHG Emissions", "attrs": {"unit": "tCO2e"}},
            {"node_id": "n_outcome", "type": "Outcome", "text": "Reduced by 10%",
             "attrs": {"direction": "decrease", "value": 10.0, "unit": "percent", "year": 2023}}
        ]

        edges = [
            {"src": "n_outcome", "dst": "n_metric", "relation": "measures"}
        ]

        evidence_sets = []
        if table_cells:
            evidence_sets.append({
                "set_id": "es_outcome",
                "attached_to_node": "n_outcome",
                "links": [{"evid_id": eid, "relation": "numeric_support"} for eid in table_cells[:2]]
            })

        if definitions:
            nodes.append({"node_id": "n_def", "type": "Definition", "text": "Scope 1+2 Definition",
                          "attrs": {"type": "standard"}})
            edges.append({"src": "n_def", "dst": "n_metric", "relation": "defines"})
            evidence_sets.append({
                "set_id": "es_def",
                "attached_to_node": "n_metric",
                "links": [{"evid_id": definitions[0], "relation": "definition_support"}]
            })

        return {
            "claim_id": "MOCK:claim:000",
            "doc_id": "MOCK",
            "nodes": nodes,
            "edges": edges,
            "evidence_sets": evidence_sets
        }


class OpenAIClient(LLMClient):
    """OpenAI GPT-4 / GPT-4o"""

    def __init__(self, model_name: str = "gpt-4o-2024-08-06", config: Dict = None):
        super().__init__(model_name, config)
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Please install: pip install openai")

    def generate(self, system_prompt: str, user_prompt: str, response_format: str = "json_object",
                 temperature: float = 0.0, max_tokens: int = 4096, **kwargs) -> Dict[str, Any]:
        start_time = time.time()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Build the request parameters
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        # Only add response_format for models that support JSON mode.
        # (Bug fix: the previous condition matched only "gpt-4", so
        # gpt-3.5-turbo-0125 never actually got response_format sent on the
        # wire, even though configs/models.yaml declared
        # response_format: json_object for it. gpt-3.5-turbo-1106 and later
        # all support JSON mode, and 0125 is among them.)
        _JSON_MODE_SUPPORTED_PREFIXES = ("gpt-4", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106")
        if response_format == "json_object" and any(
            p in self.model_name.lower() for p in _JSON_MODE_SUPPORTED_PREFIXES
        ):
            request_params["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**request_params)

        latency_ms = (time.time() - start_time) * 1000
        content = response.choices[0].message.content

        # Parse the JSON
        parsed_json = None
        if response_format == "json_object":
            try:
                parsed_json = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from OpenAI: {e}")
                parsed_json = {"error": "Invalid JSON", "raw_content": content}

        # Compute cost (GPT-4o pricing: $2.50/1M input, $10/1M output)
        usage = response.usage
        cost_usd = (usage.prompt_tokens / 1_000_000 * 2.5) + (usage.completion_tokens / 1_000_000 * 10.0)

        self.total_calls += 1
        self.total_cost += cost_usd

        return {
            "content": content,
            "parsed_json": parsed_json,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            },
            "model": self.model_name,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd
        }


class AnthropicClient(LLMClient):
    """Anthropic Claude"""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", config: Dict = None):
        super().__init__(model_name, config)
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Please install: pip install anthropic")

    def generate(self, system_prompt: str, user_prompt: str, response_format: str = "json_object",
                 temperature: float = 0.0, max_tokens: int = 4096, **kwargs) -> Dict[str, Any]:
        start_time = time.time()

        # Claude's JSON mode must be requested via the system prompt
        if response_format == "json_object":
            system_prompt += "\n\nYou must respond with valid JSON only. Do not include any text outside the JSON object."

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )

        latency_ms = (time.time() - start_time) * 1000
        content = response.content[0].text

        # Parse the JSON
        parsed_json = None
        if response_format == "json_object":
            try:
                parsed_json = json.loads(content)
            except json.JSONDecodeError:
                # Try to strip Markdown ```json ... ``` fencing (Claude
                # sometimes wraps its output this way, and sometimes also
                # appends explanatory text after the closing fence, so a
                # regex extracts only the content inside the fence rather
                # than just stripping the fence markers, to avoid trailing
                # text breaking the parse)
                fence_match = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
                cleaned = fence_match.group(1).strip() if fence_match else \
                    content.replace("```json", "").replace("```", "").strip()
                try:
                    parsed_json = json.loads(cleaned)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from Claude: {e}")
                    parsed_json = {"error": "Invalid JSON", "raw_content": content}

        # Compute cost (Claude 3.5 Sonnet: $3/1M input, $15/1M output)
        usage = response.usage
        cost_usd = (usage.input_tokens / 1_000_000 * 3.0) + (usage.output_tokens / 1_000_000 * 15.0)

        self.total_calls += 1
        self.total_cost += cost_usd

        return {
            "content": content,
            "parsed_json": parsed_json,
            "usage": {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.input_tokens + usage.output_tokens
            },
            "model": self.model_name,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd
        }


# ==========================================
# Factory function
# ==========================================
def create_llm_client(provider: str, model_name: str = None, config: Dict = None) -> LLMClient:
    """
    Create the appropriate LLM Client for the given provider

    Args:
        provider: "mock" | "openai" | "anthropic"
        model_name: model name (optional; uses the default if omitted)
        config: additional configuration

    Returns:
        An LLMClient instance
    """
    provider = provider.lower()

    if provider == "mock":
        return MockLLMClient(model_name or "mock-model", config)
    elif provider == "openai":
        return OpenAIClient(model_name or "gpt-4o-2024-08-06", config)
    elif provider == "anthropic":
        return AnthropicClient(model_name or "claude-3-5-sonnet-20241022", config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
