1. **Simple LLM Chatbot** — Call an LLM API and get responses back; your first working API integration.

2. **Basic Agent with One Tool** — An agent that can decide to call a calculator tool, see the result, and respond; the core agentic loop.

3. **Multi-Tool Agent** — Agent chooses between 2–3 tools (search, weather, files) based on the task; teaches tool routing.

4. **Simple Answer-Checker** — Script that runs an agent on test questions and checks answers against known correct ones.

5. **LLM-as-Judge Scorer** — Use a second LLM to grade the agent's answers for correctness and quality instead of exact matching.

6. **Trajectory Checker** — Logs every agent step and verifies it used the right tools in the right order, not just the final answer.

7. **Eval Report Generator** — Runs a full suite of test tasks and outputs pass rate, cost, and latency in one report.

8. **Input Filter (Prompt Injection & PII Guard)** — Screens user input for injection attempts and personal data before it reaches the agent.

9. **Output Filter (Safety Guard)** — Checks the agent's final response for unsafe, toxic, or leaked content before showing it to the user.

10. **Tool-Call Policy Enforcer** — Validates every tool call against allow/deny rules before it's allowed to execute.

11. **Agentic AI Evaluation & Guardrails Framework** — Combines all of the above into one wrapped agent with input/output/tool guardrails plus automated evaluation.