"""
LLM Reasoning Judge Module

Evaluates reasoning quality of [THINK]...[/THINK] blocks using an LLM judge.
Supports two modes:
  - API Mode: External LLM via OpenAI-compatible API (api_key or base_url set)
  - Local Mode: Model loaded via inference.py and kept resident in GPU memory

Usage:
    from utils.reasoning_judge import ReasoningJudge
    judge = ReasoningJudge()
    scores = judge.judge_batch(questions, responses, references)
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from utils.paths import load_config


# ============================================================================
# APIClient - OpenAI-compatible API wrapper
# ============================================================================

class APIClient:
    """Handles LLM calls via OpenAI-compatible API (GPT, vLLM, etc.)."""

    MAX_RETRIES = 3

    def __init__(self, api_key: str, base_url: Optional[str], model: str, timeout: int = 30):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for API mode. Install with: pip install openai"
            )

        client_kwargs = {"timeout": timeout}
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            client_kwargs["api_key"] = "no-key"
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model

    def chat(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Single chat completion with retry."""
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=256,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    print(f"[ReasoningJudge] API error after {self.MAX_RETRIES} retries: {e}")
                    return None
                time.sleep(1 * (attempt + 1))
        return None

    def cleanup(self):
        pass


# ============================================================================
# LocalClient - inference.py-based local model (resident in GPU)
# ============================================================================

class LocalClient:
    """Handles LLM calls via a locally loaded model that stays resident in GPU."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoTokenizer
        import model as model_module
        import argparse

        self.model_name = model_name

        fake_args = argparse.Namespace(
            model_type=model_name,
            model=model_name,
            local=False,
        )

        print(f"[ReasoningJudge] Loading local judge model: {model_name}")
        self.model = model_module.get_model(fake_args)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        from utils.paths import get_model_dir
        model_dir = get_model_dir()
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "train", model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        device = next(self.model.parameters()).device
        print(f"[ReasoningJudge] Judge model loaded on {device}")

    def chat(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Generate response using resident model."""
        import torch
        try:
            from inference import build_messages

            messages = build_messages([], user_prompt, system_prompt=system_prompt)

            inputs = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            if isinstance(inputs, list):
                inputs = torch.tensor([inputs])

            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            input_length = inputs.shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                )

            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"[ReasoningJudge] Local inference error: {e}")
            return None

    def cleanup(self):
        """Release model from GPU. Called at training end."""
        import torch
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[ReasoningJudge] Local judge model unloaded")


# ============================================================================
# ReasoningExtractor - Parse [THINK]...[/THINK] blocks
# ============================================================================

class ReasoningExtractor:
    """Extracts reasoning steps from [THINK]...[/THINK] blocks."""

    @staticmethod
    def extract_steps(
        text: str,
        think_start: str = "[THINK]",
        think_end: str = "[/THINK]"
    ) -> List[str]:
        """Extract reasoning steps from text between think tags."""
        pattern = re.escape(think_start) + r"(.*?)" + re.escape(think_end)
        matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            return []

        steps = []
        for block in matches:
            lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
            steps.extend(lines)

        return steps


# ============================================================================
# PromptBuilder - Constructs judge prompts
# ============================================================================

class PromptBuilder:
    """Builds prompts for the reasoning judge LLM."""

    def __init__(self):
        self.system_prompt = self._load_system_prompt()

    @staticmethod
    def _load_system_prompt() -> str:
        """Load system prompt from prompts/REASONING_JUDGE_PROMPT.txt."""
        prompt_path = os.path.join(
            os.path.dirname(__file__), "..", "prompts", "REASONING_JUDGE_PROMPT.txt"
        )
        prompt_path = os.path.normpath(prompt_path)
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"[ReasoningJudge] Warning: {prompt_path} not found, using default prompt")
            return (
                "You are a reasoning step verifier. Evaluate the logical correctness "
                "of each reasoning step. Score 0.0 to 1.0. "
                'Return ONLY a JSON: {"score": X.XX, "reason": "brief explanation"}'
            )

    def build_user_prompt(
        self,
        question: str,
        reasoning_steps: List[str],
        reference: Optional[str] = None,
    ) -> str:
        """Build user prompt for the judge."""
        steps_text = "\n".join(f"  Step {i+1}: {step}" for i, step in enumerate(reasoning_steps))

        prompt = f"Question:\n  {question}\n\nReasoning Steps:\n{steps_text}"

        if reference:
            prompt += f"\n\nReference Answer:\n  {reference}"

        return prompt


# ============================================================================
# ReasoningJudge - Orchestrator
# ============================================================================

class ReasoningJudge:
    """
    Orchestrator for LLM-based reasoning quality evaluation.

    Mode is auto-detected from config.yaml:
      - api_key or base_url set -> API mode
      - Neither set -> Local mode
      - use_vllm: true -> NotImplementedError (future)
    """

    DEFAULT_SCORE = 0.5  # Neutral fallback on errors

    def __init__(self):
        config = load_config()
        judge_config = config.get("reasoning_judge", {})

        base_url = judge_config.get("base_url", "")
        api_key = judge_config.get("api_key", "")
        model = judge_config.get("model", "biomni-r0-32b")
        use_vllm = judge_config.get("use_vllm", False)
        timeout = judge_config.get("timeout", 30)
        self.max_workers = judge_config.get("max_workers", 8)

        if use_vllm:
            raise NotImplementedError(
                "vLLM internal integration is not yet supported. "
                "Use base_url with a running vLLM server, or leave base_url empty for local mode."
            )

        if api_key or base_url:
            self.mode = "api"
            self.client = APIClient(
                api_key=api_key,
                base_url=base_url if base_url else None,
                model=model,
                timeout=timeout,
            )
            print(f"[ReasoningJudge] API mode: model={model}, base_url={base_url or 'default'}")
        else:
            self.mode = "local"
            self.client = LocalClient(model_name=model)

        self.extractor = ReasoningExtractor()
        self.prompt_builder = PromptBuilder()

    def judge_single(
        self,
        question: str,
        response_text: str,
        reference: Optional[str] = None,
        think_start: str = "[THINK]",
        think_end: str = "[/THINK]",
    ) -> float:
        """Evaluate reasoning quality for a single response."""
        steps = self.extractor.extract_steps(response_text, think_start, think_end)

        if not steps:
            return self.DEFAULT_SCORE

        user_prompt = self.prompt_builder.build_user_prompt(question, steps, reference)
        raw_response = self.client.chat(self.prompt_builder.system_prompt, user_prompt)

        if raw_response is None:
            return self.DEFAULT_SCORE

        return self._parse_score(raw_response)

    def judge_batch(
        self,
        questions: List[str],
        responses: List[str],
        references: Optional[List[str]] = None,
        think_start: str = "[THINK]",
        think_end: str = "[/THINK]",
    ) -> List[float]:
        """Evaluate reasoning quality for a batch of responses."""
        n = len(questions)
        if references is None:
            references = [None] * n

        if self.mode == "api":
            return self._judge_batch_parallel(
                questions, responses, references, think_start, think_end
            )
        else:
            return self._judge_batch_sequential(
                questions, responses, references, think_start, think_end
            )

    def _judge_batch_parallel(
        self,
        questions: List[str],
        responses: List[str],
        references: List[Optional[str]],
        think_start: str,
        think_end: str,
    ) -> List[float]:
        """API mode: parallel evaluation using ThreadPoolExecutor."""
        scores = [self.DEFAULT_SCORE] * len(questions)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {}
            for i in range(len(questions)):
                future = executor.submit(
                    self.judge_single,
                    questions[i],
                    responses[i],
                    references[i],
                    think_start,
                    think_end,
                )
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    scores[idx] = future.result()
                except Exception:
                    scores[idx] = self.DEFAULT_SCORE

        return scores

    def _judge_batch_sequential(
        self,
        questions: List[str],
        responses: List[str],
        references: List[Optional[str]],
        think_start: str,
        think_end: str,
    ) -> List[float]:
        """Local mode: sequential evaluation (model already resident)."""
        scores = []
        for i in range(len(questions)):
            score = self.judge_single(
                questions[i], responses[i], references[i], think_start, think_end
            )
            scores.append(score)
        return scores

    def cleanup(self):
        """Release resources. Called at training end."""
        if self.client is not None:
            self.client.cleanup()

    @staticmethod
    def _parse_score(raw_response: str) -> float:
        """Parse score from judge LLM response JSON."""
        try:
            match = re.search(r'\{[^}]*"score"\s*:\s*([\d.]+)[^}]*\}', raw_response)
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
        except (ValueError, AttributeError):
            pass

        try:
            data = json.loads(raw_response.strip())
            score = float(data.get("score", 0.5))
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return 0.5
