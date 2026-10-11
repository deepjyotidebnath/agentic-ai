"""
Trajectory Checker
------------------
Logs every agent step (tool call, args, output, error, latency) and verifies the
agent used the right tools in the right order, not just that the final answer was right.

Usage:
    logger = TrajectoryLogger()

    @logger.track
    def search(query): ...

    # ... run your agent, tools get logged automatically ...

    spec = TrajectorySpec(
        expected=[ExpectedStep("search", args={"query": contains("refund")}),
                  ExpectedStep("get_order"),
                  ExpectedStep("issue_refund")],
        mode="subsequence",            # "strict" | "subsequence" | "unordered"
        forbidden=["delete_account"],
        max_steps=8,
    )
    report = TrajectoryChecker(spec).check(logger.steps)
    print(report)
"""
from __future__ import annotations

import functools
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Literal


# ───────────────────────── Logging ─────────────────────────

@dataclass
class Step:
    index: int
    tool: str
    args: dict
    output: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    ts: float = field(default_factory=time.time)


class TrajectoryLogger:
    """Records every tool call in order. Works as a decorator or via manual log()."""

    def __init__(self) -> None:
        self.steps: list[Step] = []

    def log(self, tool: str, args: dict, output: Any = None,
            error: str | None = None, latency_ms: float = 0.0) -> Step:
        step = Step(len(self.steps), tool, args, output, error, latency_ms)
        self.steps.append(step)
        return step

    def track(self, fn: Callable | None = None, *, name: str | None = None):
        """Decorator: @logger.track or @logger.track(name="web_search")."""
        def deco(f: Callable):
            tool_name = name or f.__name__

            @functools.wraps(f)
            def wrapper(*a, **kw):
                args = {**{f"arg{i}": v for i, v in enumerate(a)}, **kw}
                t0 = time.perf_counter()
                try:
                    out = f(*a, **kw)
                    self.log(tool_name, args, out, None, (time.perf_counter() - t0) * 1000)
                    return out
                except Exception as e:
                    self.log(tool_name, args, None, repr(e), (time.perf_counter() - t0) * 1000)
                    raise
            return wrapper
        return deco(fn) if fn else deco

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            for s in self.steps:
                f.write(json.dumps(asdict(s), default=str) + "\n")

    @staticmethod
    def load(path: str) -> list[Step]:
        with open(path) as f:
            return [Step(**json.loads(line)) for line in f if line.strip()]


# ───────────────────────── Spec ─────────────────────────

# Arg matchers: helpers so specs can be loose about exact values.
def contains(sub: str) -> Callable[[Any], bool]:
    return lambda v: isinstance(v, str) and sub.lower() in v.lower()

def anything() -> Callable[[Any], bool]:
    return lambda v: True


@dataclass
class ExpectedStep:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)  # value or callable matcher
    optional: bool = False

    def matches(self, step: Step) -> bool:
        if step.tool != self.tool:
            return False
        for k, want in self.args.items():
            if k not in step.args:
                return False
            got = step.args[k]
            ok = want(got) if callable(want) else got == want
            if not ok:
                return False
        return True


@dataclass
class TrajectorySpec:
    expected: list[ExpectedStep]
    mode: Literal["strict", "subsequence", "unordered"] = "subsequence"
    forbidden: list[str] = field(default_factory=list)
    max_steps: int | None = None
    allow_errors: bool = False
    flag_repeats: bool = True   # identical consecutive calls (loops / wasted steps)


# ───────────────────────── Report ─────────────────────────

@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Report:
    checks: list[Check]
    precision: float   # fraction of actual steps that were expected
    recall: float      # fraction of expected (non-optional) steps that were hit
    order_score: float # LCS(expected, actual) / len(expected)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def __str__(self) -> str:
        lines = [f"TRAJECTORY {'PASS' if self.passed else 'FAIL'}  "
                 f"(precision={self.precision:.2f} recall={self.recall:.2f} order={self.order_score:.2f})"]
        for c in self.checks:
            lines.append(f"  [{'ok' if c.passed else 'XX'}] {c.name}" + (f" - {c.detail}" if c.detail else ""))
        return "\n".join(lines)


# ───────────────────────── Checker ─────────────────────────

class TrajectoryChecker:
    def __init__(self, spec: TrajectorySpec) -> None:
        self.spec = spec

    def check(self, steps: list[Step]) -> Report:
        s = self.spec
        checks: list[Check] = []

        # 1. Forbidden tools
        bad = [st.tool for st in steps if st.tool in s.forbidden]
        checks.append(Check("no forbidden tools", not bad, f"used: {bad}" if bad else ""))

        # 2. Step budget
        if s.max_steps is not None:
            checks.append(Check("within step budget", len(steps) <= s.max_steps,
                                f"{len(steps)} steps, max {s.max_steps}"))

        # 3. Errors
        if not s.allow_errors:
            errs = [f"#{st.index} {st.tool}: {st.error}" for st in steps if st.error]
            checks.append(Check("no tool errors", not errs, "; ".join(errs)))

        # 4. Repeated identical calls
        if s.flag_repeats:
            reps = [f"#{b.index} {b.tool}" for a, b in zip(steps, steps[1:])
                    if a.tool == b.tool and a.args == b.args]
            checks.append(Check("no repeated identical calls", not reps, ", ".join(reps)))

        # 5. Expected tools / order / args
        required = [e for e in s.expected if not e.optional]
        if s.mode == "strict":
            ok, detail = self._strict(steps)
        elif s.mode == "subsequence":
            ok, detail = self._subsequence(steps)
        else:
            ok, detail = self._unordered(steps, required)
        checks.append(Check(f"expected steps ({s.mode})", ok, detail))

        # Metrics
        expected_tools = [e.tool for e in s.expected]
        actual_tools = [st.tool for st in steps]
        hit_required = sum(any(e.matches(st) for st in steps) for e in required)
        recall = hit_required / len(required) if required else 1.0
        precision = (sum(t in expected_tools for t in actual_tools) / len(actual_tools)
                     if actual_tools else 0.0)
        order = self._lcs(expected_tools, actual_tools) / len(expected_tools) if expected_tools else 1.0
        return Report(checks, precision, recall, order)

    # ---- modes ----
    def _strict(self, steps: list[Step]) -> tuple[bool, str]:
        exp = self.spec.expected
        if len(steps) != len(exp):
            return False, f"expected {len(exp)} steps, got {len(steps)}: {[st.tool for st in steps]}"
        for i, (e, st) in enumerate(zip(exp, steps)):
            if not e.matches(st):
                return False, f"step {i}: expected {e.tool}{e.args or ''}, got {st.tool}{st.args}"
        return True, ""

    def _subsequence(self, steps: list[Step]) -> tuple[bool, str]:
        i = 0
        for e in self.spec.expected:
            while i < len(steps) and not e.matches(steps[i]):
                i += 1
            if i == len(steps):
                if e.optional:
                    i = 0 if False else i  # optional step missing is fine
                    continue
                return False, f"missing or out of order: {e.tool}{e.args or ''}"
            i += 1
        return True, ""

    def _unordered(self, steps: list[Step], required: list[ExpectedStep]) -> tuple[bool, str]:
        missing = [e.tool for e in required if not any(e.matches(st) for st in steps)]
        return (not missing), (f"missing: {missing}" if missing else "")

    @staticmethod
    def _lcs(a: list[str], b: list[str]) -> int:
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a)):
            for j in range(len(b)):
                dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
        return dp[-1][-1]


# ───────────────────────── Demo ─────────────────────────

if __name__ == "__main__":
    logger = TrajectoryLogger()

    @logger.track
    def search(query): return ["order #123"]

    @logger.track
    def get_order(order_id): return {"id": order_id, "total": 40}

    @logger.track
    def issue_refund(order_id, amount): return "ok"

    # Simulated agent run
    search(query="refund for my order")
    get_order(order_id=123)
    issue_refund(order_id=123, amount=40)

    spec = TrajectorySpec(
        expected=[
            ExpectedStep("search", {"query": contains("refund")}),
            ExpectedStep("get_order"),
            ExpectedStep("issue_refund", {"amount": 40}),
        ],
        mode="subsequence",
        forbidden=["delete_account"],
        max_steps=6,
    )
    print(TrajectoryChecker(spec).check(logger.steps))

    # Bad run: skips get_order, repeats a call
    bad = TrajectoryLogger()
    bad.log("search", {"query": "refund"})
    bad.log("search", {"query": "refund"})
    bad.log("issue_refund", {"amount": 40})
    print()
    print(TrajectoryChecker(spec).check(bad.steps))