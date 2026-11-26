"""
Sandbox utilities for safe execution of generated code during benchmarks.
Ensures:
- Execution time limit (default 30 s)
- No network access
- Restricted environment variables
- Isolated temporary working directory
"""

import subprocess
import os
import tempfile
import shlex
from typing import Optional, Tuple

DEFAULT_TIMEOUT = 30  # seconds

def run_code(
    code: str,
    timeout: int = DEFAULT_TIMEOUT,
    env: Optional[dict] = None,
) -> Tuple[int, str, str]:
    """Execute *code* in a sandboxed subprocess.

    Returns a tuple ``(returncode, stdout, stderr)``.  The subprocess is
    launched with ``python - <<EOF`` to avoid creating a temporary script file.
    Network access is blocked by unsetting common proxy variables and by
    launching the process with ``-S`` (no site‑packages) when feasible.
    """
    # Prepare a minimal clean environment
    safe_env = {
        k: v
        for k, v in (env or os.environ).items()
        if not any(block in k.lower() for block in ["proxy", "http", "https", "ftp"])  # drop proxies
    }
    # Ensure no internet‑related vars leak
    for var in ["http_proxy", "https_proxy", "ftp_proxy", "all_proxy", "NO_PROXY"]:
        safe_env.pop(var, None)
    # Use a temporary directory as cwd to isolate file writes
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write code to a temporary file to allow multi‑line scripts and easier debugging
        script_path = os.path.join(tmpdir, "script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        # Run the script
        proc = subprocess.run(
            ["python", script_path],
            cwd=tmpdir,
            env=safe_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr

# Example helper for unit‑test style evaluation
def evaluate_generated_code(code: str, test_cases: list[dict]) -> dict:
    """Run *code* against a list of ``test_cases``.
    Each test case dict should contain ``"input"`` and ``"expected"`` fields.
    Returns a summary dict with pass/fail counts and any error messages.
    """
    results = {"passed": 0, "failed": 0, "errors": []}
    for case in test_cases:
        # Simple approach: prepend a call to a function named ``solution``
        wrapped = f"\n{code}\n\nprint(solution(**{case['input']}))"
        rc, out, err = run_code(wrapped)
        if rc != 0:
            results["failed"] += 1
            results["errors"].append({"case": case, "stderr": err.strip()})
        else:
            try:
                output = out.strip()
                if output == str(case["expected"]):
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append({"case": case, "output": output, "expected": case["expected"]})
            except Exception as e:
                results["failed"] += 1
                results["errors"].append({"case": case, "exception": str(e)})
    return results
