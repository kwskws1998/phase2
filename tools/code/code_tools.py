"""
Code generation tools.
Uses LLM to generate code based on task description.
Includes auto-execution and error correction loop.
"""

import os
from typing import Dict, Any
from tools.base import Tool, ToolParameter, register_tool
from tools.biomni.bio_tools import generate_with_llm

MAX_CODE_FIX_ATTEMPTS = 3


def load_code_gen_prompt():
    """Load CODE_GEN_PROMPT.txt if available."""
    prompt_path = os.path.join(os.path.dirname(__file__), "../../prompts/CODE_GEN_PROMPT.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return None


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


@register_tool
class CodeGenTool(Tool):
    """Generate code based on task description."""
    
    name = "code_gen"
    description = "Generate code based on task description. Supports Python and R."
    parameters = [
        ToolParameter(
            name="task",
            type="string",
            description="What the code should do (detailed description)",
            required=True
        ),
        ToolParameter(
            name="language",
            type="string",
            description="Programming language (python or r)",
            required=False,
            default="python",
            enum=["python", "r"]
        ),
        ToolParameter(
            name="context",
            type="string",
            description="Additional context, requirements, or constraints",
            required=False,
            default=""
        )
    ]
    
    def execute(self, task: str = None, language: str = "python", context: str = "", **kwargs) -> Dict[str, Any]:
        """Generate code, execute it, and auto-fix errors up to MAX_CODE_FIX_ATTEMPTS times.
        
        Args:
            task: Description of what the code should do
            language: Programming language (default: python)
            context: Additional context or requirements
            **kwargs: conv_id, step_index (injected by plan loop), step_description, description
            
        Returns:
            Dict with success status, generated code, and execution results
        """
        language = language.lower()
        if language not in ("python", "r"):
            return {
                "success": False,
                "error": f"Unsupported language: {language}. Only 'python' and 'r' are supported.",
                "result": None
            }

        if not task:
            task = kwargs.get('step_description', '') or kwargs.get('description', '')
            if not task:
                task = "Generate data analysis and visualization code."
        
        system_prompt = load_code_gen_prompt()
        
        prompt = f"Write {language} code to accomplish the following task:\n\n{task}"
        if context:
            prompt += f"\n\nAdditional context/requirements:\n{context}"
        prompt += "\n\nOutput ONLY the code. No explanations, no markdown code blocks, just the raw code."
        
        print(f"[code_gen] Generating {language} code, prompt length: {len(prompt)} chars")
        code = generate_with_llm(prompt, max_tokens=1500, system_prompt=system_prompt)
        
        if not code:
            print(f"[code_gen] generate_with_llm returned empty string")
            return {
                "success": False,
                "error": "LLM returned empty output - model may be busy or context too large",
                "result": None
            }
        
        code = _strip_markdown_fences(code)
        
        if not code:
            print(f"[code_gen] Code empty after cleanup, returning raw output")
            return {
                "success": False,
                "error": "Generated output contained no usable code",
                "result": None
            }
        
        print(f"[code_gen] Successfully generated {len(code)} chars of {language} code")

        conv_id = kwargs.get('conv_id')
        step_index = kwargs.get('step_index')

        if conv_id is None:
            return {
                "success": True,
                "result": {
                    "language": language,
                    "code": code,
                    "task": task
                }
            }

        from inference import _execute_code_subprocess

        exec_result = None
        fix_attempts = 0

        for attempt in range(MAX_CODE_FIX_ATTEMPTS + 1):
            print(f"[code_gen] Executing code (attempt {attempt + 1}/{MAX_CODE_FIX_ATTEMPTS + 1})")
            exec_result = _execute_code_subprocess(code, language, conv_id, step_index)

            if exec_result['success'] or not exec_result.get('stderr', '').strip():
                fix_attempts = attempt
                print(f"[code_gen] Code executed successfully (after {attempt} fix(es))")
                break

            print(f"[code_gen] Execution error: {exec_result['stderr'][:200]}")

            if attempt < MAX_CODE_FIX_ATTEMPTS:
                fix_prompt = (
                    f"The following {language} code produced an error when executed.\n\n"
                    f"--- CODE ---\n{code}\n--- END CODE ---\n\n"
                    f"--- ERROR ---\n{exec_result['stderr']}\n--- END ERROR ---\n\n"
                    f"Fix the code so it runs without errors. "
                    f"Output ONLY the corrected code. No explanations, no markdown code blocks."
                )
                print(f"[code_gen] Requesting LLM fix (attempt {attempt + 1}/{MAX_CODE_FIX_ATTEMPTS})")
                fixed_code = generate_with_llm(fix_prompt, max_tokens=1500, system_prompt=system_prompt)

                if fixed_code:
                    fixed_code = _strip_markdown_fences(fixed_code)
                    if fixed_code:
                        code = fixed_code
                        print(f"[code_gen] Got fixed code ({len(code)} chars)")
                        continue

                print(f"[code_gen] LLM returned empty fix, stopping retry loop")
                fix_attempts = attempt + 1
                break
            else:
                fix_attempts = MAX_CODE_FIX_ATTEMPTS

        return {
            "success": True,
            "result": {
                "language": language,
                "code": code,
                "task": task,
                "execution": exec_result,
                "fix_attempts": fix_attempts
            }
        }
