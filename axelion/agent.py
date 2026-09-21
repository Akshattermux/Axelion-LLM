"""
Axelion Agent & Tool-Calling Engine.
Enables Axelion to reason, call external Python tools/APIs autonomously,
and synthesize answers in a multi-turn agent loop.
"""

import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Union

from .model import Axelion
from .tokenizer import AxelionTokenizer
from .generation import generate_text


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters_schema: Dict[str, Any]

    def execute(self, **kwargs) -> Any:
        try:
            return self.func(**kwargs)
        except Exception as e:
            return f"Error executing tool '{self.name}': {str(e)}"


class ToolRegistry:
    """Registry of tools available to the agent."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        t_name = name or func.__name__
        doc = description or (func.__doc__ or "").strip() or f"Execute {t_name}"
        sig = inspect.signature(func)
        properties = {}
        required = []

        for p_name, p in sig.parameters.items():
            param_type = "string"
            if p.annotation == int:
                param_type = "integer"
            elif p.annotation == float:
                param_type = "number"
            elif p.annotation == bool:
                param_type = "boolean"
            elif p.annotation == list:
                param_type = "array"
            elif p.annotation == dict:
                param_type = "object"

            properties[p_name] = {"type": param_type, "description": f"Argument {p_name}"}
            if p.default == inspect.Parameter.empty:
                required.append(p_name)

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        self.tools[t_name] = Tool(
            name=t_name,
            description=doc,
            func=func,
            parameters_schema=schema,
        )

    def tool(self, name: Optional[str] = None, description: Optional[str] = None):
        """Decorator to register a tool function."""
        def decorator(func: Callable):
            self.register(func, name=name, description=description)
            return func
        return decorator

    def get_tool_descriptions(self) -> str:
        lines = []
        for t in self.tools.values():
            params = ", ".join(f"{k}: {v['type']}" for k, v in t.parameters_schema["properties"].items())
            lines.append(f"- `{t.name}({params})`: {t.description}")
        return "\n".join(lines)


@dataclass
class AgentStep:
    thought_or_call: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None


class AxelionAgent:
    """
    Autonomous multi-turn agent driven by Axelion LLM.
    Supports tool execution, structured JSON action generation, and reasoning.
    """

    SYSTEM_TEMPLATE = """You are Axelion Agent, an autonomous assistant with access to tools.
When given a task, you can call tools to fetch information or perform calculations.

Available Tools:
{tools}

Format for calling a tool:
```json
{{"action": "tool_name", "parameters": {{"arg1": "value1"}}}}
```

When you have the final answer or do not require any tools, output your final answer directly to the user without any json block.
"""

    def __init__(
        self,
        model: Axelion,
        tokenizer: AxelionTokenizer,
        registry: Optional[ToolRegistry] = None,
        max_steps: int = 5,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry or ToolRegistry()
        self.max_steps = max_steps
        self.device = device

    def _extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        # Look for ```json { ... } ``` or raw JSON
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = json_match.group(1) if json_match else None
        if not candidate:
            candidate_match = re.search(r'(\{\s*"action"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{.*?\}\s*\})', text, re.DOTALL)
            candidate = candidate_match.group(1) if candidate_match else None

        if candidate:
            try:
                data = json.loads(candidate)
                if "action" in data:
                    return data
            except json.JSONDecodeError:
                pass
        return None

    def run(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """Execute the agent loop on the query."""
        tools_str = self.registry.get_tool_descriptions()
        system_prompt = self.SYSTEM_TEMPLATE.format(tools=tools_str)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        steps: List[AgentStep] = []

        if verbose:
            print(f"\n[Agent Started] Query: {query}")

        for step_idx in range(self.max_steps):
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            response = generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_new_tokens=256,
                temperature=0.2,  # Low temperature for deterministic tool invocation
                device=self.device,
            ).strip()

            tool_call = self._extract_tool_call(response)

            if not tool_call:
                # Agent provided final answer
                if verbose:
                    print(f"\n[Agent Final Answer]:\n{response}\n")
                return {
                    "final_answer": response,
                    "steps": steps,
                    "completed": True,
                }

            action = tool_call.get("action")
            params = tool_call.get("parameters", {})

            if verbose:
                print(f"[Step {step_idx + 1}] Invoking tool: {action} with {params}")

            if action in self.registry.tools:
                tool = self.registry.tools[action]
                observation = str(tool.execute(**params))
            else:
                observation = f"Tool '{action}' not found in registry."

            if verbose:
                print(f"[Observation]: {observation}")

            steps.append(
                AgentStep(
                    thought_or_call=response,
                    tool_name=action,
                    tool_args=params,
                    observation=observation,
                )
            )

            # Add assistant action and tool observation back to conversation history
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {observation}\nContinue to answer or call next tool."})

        # If loop finishes without explicit final answer
        final_answer = steps[-1].observation if steps else "Max iterations reached without answer."
        return {
            "final_answer": final_answer,
            "steps": steps,
            "completed": False,
        }
