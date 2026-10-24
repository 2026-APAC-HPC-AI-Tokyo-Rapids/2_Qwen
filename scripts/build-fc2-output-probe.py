"""Extract actual MegaMoE FC2 epilogue and combine bodies for a one-GPU check."""
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract(text):
    epilogue = text.split('// ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------\n', 1)[1]
    epilogue = epilogue.split('\n            }\n        };', 1)[0]
    combine = text.split('        constexpr uint32_t kNumHiddenBytes =', 1)[1]
    combine = '        constexpr uint32_t kNumHiddenBytes =' + combine.split('\n    }\n#else', 1)[0]
    helpers = ''
    if '    // All Qwen ranks use the same 32-column permutation' in text:
        helpers = text.split('    // All Qwen ranks use the same 32-column permutation', 1)[1]
        helpers = helpers.split('    // =====================================================================', 1)[0]
        helpers = '    // All Qwen ranks use the same 32-column permutation' + helpers
    return helpers, epilogue, combine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'output'):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    base, cand = a.baseline.read_text(), a.candidate.read_text()
    # A probe that accidentally omits a changed math/sync body must fail closed.
    def outside(text):
        for part in extract(text):
            if part:
                text = text.replace(part, '', 1)
        return text
    assert outside(base) == outside(cand), 'Candidate changes code outside FC2 output/combine probe'
    template = (ROOT / 'kernels/probes/fc2_output.cu.in').read_text()
    prefix, kernel, suffix = template.split('// EXTRACTED_KERNEL\n')
    functions = []
    for name, text in [('baseline', base), ('candidate', cand)]:
        helpers, epilogue, combine = extract(text)
        functions.append(kernel.replace('@NAME@', name).replace('@HELPERS@', helpers).replace('@EPILOGUE@', epilogue).replace('@COMBINE@', combine))
        print(name, hashlib.sha256(text.encode()).hexdigest())
    a.output.write_text(prefix + '\n'.join(functions) + suffix)


if __name__ == '__main__':
    main()
