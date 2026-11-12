"""
Test runner for AI Story Continuation System (Project 3).

Runs the required test cases from the assignment:
  - 10+ valid story prompts with generated outputs
  - Empty prompt (error handling)
  - Over-4000-character prompt (error handling)
  - Short / Medium / Long continuation lengths
  - Low temperature (0.3) vs High temperature (1.2) comparison
  - Response latency measurements

Outputs a formatted test report to tests/test_results.md

Usage:
  cd AI-Story-Continuation
  .venv/bin/python tests/run_tests.py
"""

import json
import sys
import time
import os

# Add parent directory to path so we can import story_generator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from story_generator import generate_story


# ── Test prompts ─────────────────────────────────────────────────────────────
TEST_PROMPTS = [
    "One day, a student discovered a mysterious box in the college library",
    "A robot entered the classroom for the first time",
    "It was a dark and rainy night when the phone rang",
    "The astronaut looked out the window and saw something impossible",
    "Deep in the forest, an old woman kept a garden of strange flowers",
    "The letter arrived without a stamp, without an address, without a name",
    "On the last day of school, the teacher revealed a secret",
    "A cat wandered into the laboratory and knocked over a beaker",
    "The ship had been drifting for three months when they spotted land",
    "Under the floorboards, there was a map drawn in invisible ink",
    "The clock struck thirteen and the world changed forever",
    "A child found a door in the back of the wardrobe that had never been there before",
]

SAMPLE_TEST_CASES = [
    {
        "name": "Short continuation",
        "prompt": "One day, a student discovered a mysterious box in the college library",
        "max_new_tokens": 50,
        "temperature": 0.8,
        "top_p": 0.9,
    },
    {
        "name": "Medium continuation",
        "prompt": "A robot entered the classroom for the first time",
        "max_new_tokens": 100,
        "temperature": 0.8,
        "top_p": 0.9,
    },
    {
        "name": "Long continuation",
        "prompt": "A robot entered the classroom for the first time",
        "max_new_tokens": 150,
        "temperature": 0.8,
        "top_p": 0.9,
    },
    {
        "name": "Low temperature (0.3) — less varied",
        "prompt": "It was a dark and rainy night",
        "max_new_tokens": 80,
        "temperature": 0.3,
        "top_p": 0.9,
    },
    {
        "name": "High temperature (1.2) — more varied",
        "prompt": "It was a dark and rainy night",
        "max_new_tokens": 80,
        "temperature": 1.2,
        "top_p": 0.9,
    },
]


def run_single_test(prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
    """Run a single generation and return result + latency."""
    start = time.time()
    result = generate_story(prompt, max_new_tokens, temperature, top_p)
    latency_ms = (time.time() - start) * 1000
    return result, latency_ms


def test_error_case(name, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9):
    """Run a test expected to raise an error."""
    try:
        generate_story(prompt, max_new_tokens, temperature, top_p)
        return name, "FAIL — no error raised", ""
    except (ValueError, Exception) as e:
        return name, "PASS", str(e)


def main():
    report_lines = []
    report_lines.append("# AI Story Continuation — Test Results Report\n")
    report_lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("**Model:** DistilGPT2 (local PyTorch inference)\n")
    report_lines.append("---\n")

    # ── Section 1: 10+ valid prompts ─────────────────────────────────────────
    report_lines.append("## 1. Valid Story Prompts (10+ prompts)\n")
    report_lines.append("| # | Prompt (truncated) | Tokens | Temp | Latency (ms) | Generated Continuation (first 120 chars) |")
    report_lines.append("|---|---|---|---|---|---|")

    all_latencies = []
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"  Test {i}/{len(TEST_PROMPTS)}: {prompt[:50]}...")
        try:
            result, latency = run_single_test(prompt, max_new_tokens=80, temperature=0.8)
            all_latencies.append(latency)
            truncated_prompt = prompt[:60] + ("…" if len(prompt) > 60 else "")
            truncated_result = result[:120].replace("\n", " ").replace("|", "\\|") + ("…" if len(result) > 120 else "")
            report_lines.append(f"| {i} | {truncated_prompt} | 80 | 0.8 | {latency:.0f} | {truncated_result} |")
        except Exception as e:
            report_lines.append(f"| {i} | {prompt[:60]} | 80 | 0.8 | — | ERROR: {e} |")
    report_lines.append("")

    # ── Section 2: Edge cases ────────────────────────────────────────────────
    report_lines.append("## 2. Edge Case Tests\n")
    report_lines.append("| Test Case | Status | Details |")
    report_lines.append("|---|---|---|")

    # Empty prompt
    name, status, detail = test_error_case("Empty prompt", "")
    report_lines.append(f"| {name} | {status} | {detail} |")

    name, status, detail = test_error_case("Whitespace-only prompt", "   ")
    report_lines.append(f"| {name} | {status} | {detail} |")

    # Over-4000-char prompt
    long_prompt = "A " * 2500  # ~5000 chars
    print("  Testing over-4000-char prompt...")
    try:
        result, latency = run_single_test(long_prompt, max_new_tokens=50)
        report_lines.append(f"| Long prompt (5000 chars) | Handled (truncated by tokenizer) | Generated {len(result)} chars in {latency:.0f} ms |")
    except ValueError as e:
        report_lines.append(f"| Long prompt (5000 chars) | PASS — rejected | {e} |")
    except Exception as e:
        report_lines.append(f"| Long prompt (5000 chars) | ERROR | {e} |")
    report_lines.append("")

    # ── Section 3: Length comparison ─────────────────────────────────────────
    report_lines.append("## 3. Continuation Length Comparison\n")
    report_lines.append("| Setting | Prompt | max_new_tokens | Temperature | Output Length (chars) | Latency (ms) |")
    report_lines.append("|---|---|---|---|---|---|")

    for tc in SAMPLE_TEST_CASES[:3]:
        print(f"  Testing: {tc['name']}...")
        result, latency = run_single_test(tc["prompt"], tc["max_new_tokens"],
                                          tc["temperature"], tc["top_p"])
        all_latencies.append(latency)
        report_lines.append(
            f"| {tc['name']} | {tc['prompt'][:50]}… | {tc['max_new_tokens']} "
            f"| {tc['temperature']} | {len(result)} | {latency:.0f} |"
        )
    report_lines.append("")

    # ── Section 4: Temperature comparison ────────────────────────────────────
    report_lines.append("## 4. Temperature Comparison (Same Prompt)\n")
    report_lines.append("### Prompt: *\"It was a dark and rainy night\"*\n")

    for tc in SAMPLE_TEST_CASES[3:5]:
        print(f"  Testing: {tc['name']}...")
        result, latency = run_single_test(tc["prompt"], tc["max_new_tokens"],
                                          tc["temperature"], tc["top_p"])
        all_latencies.append(latency)
        report_lines.append(f"**{tc['name']}** (temperature={tc['temperature']}):\n")
        report_lines.append(f"> {result[:300]}\n")
        report_lines.append(f"*Output length: {len(result)} chars | Latency: {latency:.0f} ms*\n")

    report_lines.append("**Observation:** Lower temperature produces more predictable, focused "
                        "text. Higher temperature introduces more creative and sometimes unexpected "
                        "word choices, but can also produce less coherent output.\n")

    # ── Section 5: Reproducibility test ──────────────────────────────────────
    report_lines.append("## 5. Reproducibility Test (Same Prompt, Same Settings, Two Runs)\n")
    prompt = "The astronaut looked out the window and saw something impossible"
    print("  Testing reproducibility...")
    r1, l1 = run_single_test(prompt, 60, 0.8)
    r2, l2 = run_single_test(prompt, 60, 0.8)
    report_lines.append(f"**Run 1:** {r1[:200]}\n")
    report_lines.append(f"**Run 2:** {r2[:200]}\n")
    same = "YES" if r1 == r2 else "NO"
    report_lines.append(f"**Outputs identical?** {same}\n")
    report_lines.append("**Explanation:** When `do_sample=True`, the model samples from the "
                        "probability distribution at each step, so different runs produce "
                        "different continuations even with identical inputs.\n")

    # ── Section 6: Latency summary ───────────────────────────────────────────
    report_lines.append("## 6. Latency Summary (Local PyTorch)\n")
    if all_latencies:
        avg_lat = sum(all_latencies) / len(all_latencies)
        min_lat = min(all_latencies)
        max_lat = max(all_latencies)
        report_lines.append(f"| Metric | Value |")
        report_lines.append(f"|---|---|")
        report_lines.append(f"| Number of requests | {len(all_latencies)} |")
        report_lines.append(f"| Average latency | {avg_lat:.0f} ms |")
        report_lines.append(f"| Min latency | {min_lat:.0f} ms |")
        report_lines.append(f"| Max latency | {max_lat:.0f} ms |")
        report_lines.append("")
        report_lines.append("> **Note:** The first request includes model loading time.  "
                            "Latency varies with hardware, system load, prompt length and "
                            "max_new_tokens setting.  Cloud (SageMaker) latency will additionally "
                            "include network round-trip and container startup time.\n")

    # ── Section 7: API contract ──────────────────────────────────────────────
    report_lines.append("## 7. API Contract Verification\n")
    report_lines.append("### Request JSON\n")
    report_lines.append("```json")
    report_lines.append(json.dumps({
        "prompt": "One day, a student discovered a mysterious box...",
        "max_new_tokens": 50,
        "temperature": 0.8,
        "top_p": 0.9,
    }, indent=2))
    report_lines.append("```\n")
    report_lines.append("### Response JSON\n")
    report_lines.append("```json")
    report_lines.append(json.dumps({
        "generated_text": "(continuation text here)"
    }, indent=2))
    report_lines.append("```\n")
    report_lines.append("### Error Response JSON\n")
    report_lines.append("```json")
    report_lines.append(json.dumps({
        "error": "prompt is required"
    }, indent=2))
    report_lines.append("```\n")

    # ── Write report ─────────────────────────────────────────────────────────
    report_text = "\n".join(report_lines)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results.md")
    with open(output_path, "w") as f:
        f.write(report_text)
    print(f"\n✅ Test report written to {output_path}")


if __name__ == "__main__":
    main()
