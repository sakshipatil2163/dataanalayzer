"""
Run this script locally or inside CI to execute tests
on all 3 datasets and generate an HTML test report.
Usage: python ci_cd_runner.py
"""
import subprocess, sys

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "tests/test_analyzer.py",
        "-v",
        "--html=test_report.html",
        "--self-contained-html"
    ],
    capture_output=True,
    text=True
)
print(result.stdout)
print(result.stderr)
sys.exit(result.returncode)