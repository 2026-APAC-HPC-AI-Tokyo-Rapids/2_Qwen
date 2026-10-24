"""Count static instructions in nvdisasm -c -g -hex output (not executed cycles)."""
import argparse
import collections
import json
import re
from pathlib import Path


def geometry(symbol):
    if 'ELj128ELj256ELj128ELj141312' in symbol:
        return 'prefill140' if 'ELj140ELj8' in symbol else 'prefill128'
    if 'ELj140ELj8' in symbol:
        return 'masked_nonswap'
    if 'ELj8ELj16ELj64' in symbol:
        return 'partial_wave'
    return 'decode_independent' if 'ELj2260992ELj6' in symbol else 'decode_shared'


def inspect(path, output):
    result = {}
    sections = re.split(r'(?m)^\s*\.section\s+\.text\.', path.read_text())[1:]
    for section in sections:
        symbol = section.split(',', 1)[0]
        if not symbol.startswith('_ZN9deep_gemm22sm90_fp8_mega_moe_impl'):
            continue
        name = geometry(symbol)
        assert name not in result, f'Duplicate geometry: {name}'
        location, compact, encodings = '', [], []
        counts = collections.Counter()
        for line in section.splitlines():
            match = re.search(r'//## File "(.+)", line (\d+)', line)
            if match:
                location = Path(match[1]).name + ':' + match[2]
            match = re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.+?;)\s*/\* (0x[0-9a-f]+) \*/', line)
            if match:
                address, instruction, encoding = match.groups()
                words = instruction.split()
                counts[(words[1] if words[0].startswith('@') else words[0]).rstrip(';')] += 1
                compact.append(f'{address}  {instruction:85} {location}')
                encodings.append(encoding)
            elif compact:
                # The second 64-bit word contains the remaining encoding/control.
                match = re.fullmatch(r'\s*/\* (0x[0-9a-f]+) \*/\s*', line)
                if match:
                    encodings.append(match[1])
        assert compact, f'No instructions: {name}'
        output.mkdir(parents=True, exist_ok=True)
        (output / f'{name}.sass').write_text('\n'.join(compact) + '\n')
        (output / f'{name}.encoding').write_text('\n'.join(encodings) + '\n')
        result[name] = {'instructions_including_padding': len(compact),
                        'opcodes': dict(sorted(counts.items()))}
    assert len(result) == 6, 'Expected the six full-kernel probe geometries'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = {name: inspect(path, args.output / name)
              for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]}
    (args.output / 'counts.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, before in result['baseline'].items():
        after = result['candidate'][name]
        identical = ((args.output / 'baseline' / f'{name}.encoding').read_bytes() ==
                     (args.output / 'candidate' / f'{name}.encoding').read_bytes())
        print(name, before['instructions_including_padding'], '->',
              after['instructions_including_padding'], 'identical encoding:', identical)
        print({op: [before['opcodes'].get(op, 0), after['opcodes'].get(op, 0)]
               for op in sorted(before['opcodes'].keys() | after['opcodes'].keys())
               if before['opcodes'].get(op, 0) != after['opcodes'].get(op, 0)})


if __name__ == '__main__':
    main()
