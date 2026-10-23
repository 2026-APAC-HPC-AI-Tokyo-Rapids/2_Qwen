"""Generate a one-GPU FC1 probe from the actual control/candidate header loops."""
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def between(text, start, end):
    a = text.index(start)
    return text[a:text.index(end, a)]


def fc1(text):
    text = text[text.index("constexpr bool kQwenPrefillMainloop ="):]
    return between(text, "                        const uint32_t gate_row =", "                    } else {\n                        // Per-128 activation scales:")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    baseline, candidate = args.baseline.read_text(), args.candidate.read_text()
    # This probe extracts FC1 only; do not silently ignore changes elsewhere.
    if baseline.replace(fc1(baseline), "FC1", 1) != candidate.replace(fc1(candidate), "FC1", 1):
        raise ValueError("This probe supports changes inside the specialized FC1 body only")
    source = (ROOT / "kernels/probes/prefill_fc1.cu.in").read_text()
    prefix, kernel, suffix = source.split("// EXTRACTED_KERNEL\n")
    functions = []
    for name, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        text = path.read_text()
        post = between(text, "                auto postscale_l1_final =", "                auto prescale_l2_final =")
        issue = between(text, "                    auto issue =", "                    if (block_phase == sched::BlockPhase::Linear1) {")
        functions.append(kernel.replace("@NAME@", name).replace("@POSTSCALE@", post).replace("@ISSUE@", issue).replace("@LOOP@", fc1(text)))
        print(name, hashlib.sha256(path.read_bytes()).hexdigest())
    args.output.write_text(prefix + "\n".join(functions) + suffix)


if __name__ == "__main__":
    main()
