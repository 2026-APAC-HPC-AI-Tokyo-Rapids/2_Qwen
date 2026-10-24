"""Generate one-GPU FC1 and FC2 probes from the serving header math loops."""
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


def fc2(text):
    text = text[text.index("// Per-128 activation scales: one WGMMA group per K block"):]
    return between(text, "                        // The warpgroup's 256 output columns", "                        } else {\n                        const uint32_t l2_row =")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    baseline, candidate = args.baseline.read_text(), args.candidate.read_text()
    # Do not silently ignore changes outside the measured math bodies.
    def outside(text):
        return text.replace(fc1(text), "FC1", 1).replace(fc2(text), "FC2", 1)
    if outside(baseline) != outside(candidate):
        raise ValueError("This probe supports changes inside the specialized FC1/FC2 bodies only")
    source = (ROOT / "kernels/probes/prefill_math.cu.in").read_text()
    prefix, kernel, suffix = source.split("// EXTRACTED_KERNEL\n")
    functions = []
    for name, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        text = path.read_text()
        post = between(text, "                auto postscale_l1_final =", "                auto prescale_l2_final =")
        issue = between(text, "                    auto issue =", "                    if (block_phase == sched::BlockPhase::Linear1) {")
        for phase, extract, k, n in (("fc1", fc1, 4096, 3072), ("fc2", fc2, 1536, 4096)):
            functions.append(kernel.replace("@NAME@", name+"_"+phase).replace("@K@", str(k)).replace("@N@", str(n)).replace("@POSTSCALE@", post).replace("@ISSUE@", issue).replace("@LOOP@", extract(text)))
        print(name, hashlib.sha256(path.read_bytes()).hexdigest())
    args.output.write_text(prefix + "\n".join(functions) + suffix)


if __name__ == "__main__":
    main()
