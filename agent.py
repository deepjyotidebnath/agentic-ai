"""A simple AI agent: Groq LLM + tools in a loop.

Setup:
    pip install groq python-dotenv
    Put your key in a .env file:  GROQ_API_KEY=your-key
    (get one at https://console.groq.com)
    python agent.py
"""

import ast
import json
import operator
import os
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # loads variables from .env into the environment

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY not found. Add it to your .env file.")

client = Groq(api_key=api_key)
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# ---------- Tools ----------

def calculator(expression: str) -> str:
    """Safely evaluate a basic math expression."""
    ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
    }

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("Unsupported expression")

    try:
        return str(ev(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


def get_time() -> str:
    return datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")


TOOL_FUNCTIONS = {
    "calculator": lambda args: calculator(args["expression"]),
    "get_time": lambda args: get_time(),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression, e.g. '(12 + 8) * 3'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------- Agent loop ----------

def run_agent(user_input: str, history: list) -> str:
    history.append({"role": "user", "content": user_input})

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1024,
        )
        msg = response.choices[0].message

        # No tool requested -> we have the final answer
        if not msg.tool_calls:
            history.append({"role": "assistant", "content": msg.content})
            return msg.content

        # Record the assistant's tool request
        history.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # Run each tool and send the results back
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            print(f"  [tool] {tc.function.name}({args})")
            output = TOOL_FUNCTIONS[tc.function.name](args)
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })


if __name__ == "__main__":
    history = [{
        "role": "system",
        "content": "You are a helpful assistant. Use tools when they help.",
    }]
    print("Agent ready. Type 'quit' to exit.")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        print("Agent:", run_agent(text, history))