"""Streamlit chat UI for the Groq agent.

Setup:
    pip install streamlit groq python-dotenv
    (keep agent.py and .env in the same folder)
    streamlit run app.py
"""

import json

import streamlit as st

try:
    from agent import MODEL, TOOL_FUNCTIONS, TOOLS, client
except SystemExit as e:  # raised by agent.py when GROQ_API_KEY is missing
    st.error(str(e))
    st.stop()

st.set_page_config(page_title="Groq Agent", page_icon="🤖")
st.title("🤖 Groq Agent")
st.caption(f"Model: {MODEL}")

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful assistant. Use tools when they help.",
}

# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state.history = [SYSTEM_PROMPT]  # what the model sees
if "display" not in st.session_state:
    st.session_state.display = []  # what the UI shows

with st.sidebar:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = [SYSTEM_PROMPT]
        st.session_state.display = []
        st.rerun()
    st.markdown("**Tools**")
    for t in TOOLS:
        st.markdown(f"- `{t['function']['name']}`: {t['function']['description']}")


def run_agent(user_input: str):
    """Run the agent loop. Returns (answer, list_of_tool_calls_made)."""
    history = st.session_state.history
    history.append({"role": "user", "content": user_input})
    tool_log = []

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1024,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            history.append({"role": "assistant", "content": msg.content})
            return msg.content, tool_log

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

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            output = TOOL_FUNCTIONS[tc.function.name](args)
            tool_log.append({"name": tc.function.name, "args": args, "output": output})
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": output,
            })


def render_tools(tool_log):
    for call in tool_log:
        with st.expander(f"🔧 {call['name']}", expanded=False):
            st.write("**Input:**", call["args"])
            st.write("**Output:**", call["output"])


# ---------- Render past messages ----------
for m in st.session_state.display:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_tools(m.get("tools", []))

# ---------- Handle new input ----------
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.display.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, tool_log = run_agent(prompt)
            except Exception as e:
                answer, tool_log = f"Error: {e}", []
        st.markdown(answer)
        render_tools(tool_log)

    st.session_state.display.append(
        {"role": "assistant", "content": answer, "tools": tool_log}
    )
        
