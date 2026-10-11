import ast
import json
import operator
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()
MODEL = "openai/gpt-oss-120b"
MAX_STEPS = 10

# ---------- Tools ----------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}

def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")

def calculator(expression: str) -> str:
    return str(_eval(ast.parse(expression, mode="eval").body))

def get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

TOOL_FUNCS = {"calculator": calculator, "get_time": get_time}

TOOLS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '(12 + 8) * 3 / 4'.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_time",
        "description": "Get the current local date and time.",
        "parameters": {"type": "object", "properties": {}}}},
]

# ---------- Agent loop ----------
def run_agent(user_input: str, history: list) -> str:
    history.append({"role": "user", "content": user_input})

    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL, messages=history, tools=TOOLS,
            tool_choice="auto", max_tokens=1024,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            history.append({"role": "assistant", "content": msg.content})
            return msg.content or ""

        history.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            print(f"  [tool] {tc.function.name}({args})")
            try:
                output = TOOL_FUNCS[tc.function.name](**args)
            except Exception as e:
                output = f"Error: {e}"
            history.append({"role": "tool", "tool_call_id": tc.id,
                            "content": str(output)})

    return "Stopped: reached max steps."


if __name__ == "__main__":
    history = [{"role": "system",
                "content": "You are a helpful assistant. Use tools when they help."}]
    print("Agent ready. Type 'quit' to exit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        print("Agent:", run_agent(q, history))