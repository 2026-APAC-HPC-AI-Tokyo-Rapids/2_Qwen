"""Static instruction census. Address ranges are explicit, never dynamic counts."""
import re
import sys
from pathlib import Path

OPS = ('LDL', 'STL', 'MUFU.EX2', 'F2FP', 'FMNMX', 'FFMA', 'FADD', 'FMUL', 'PRMT', 'SHFL', 'BAR', 'WARPGROUP.DEPBAR')


def counts(instructions):
    return {op: sum(mnemonic == op or mnemonic.startswith(op + '.') for _, mnemonic, _ in instructions) for op in OPS}



def softmax_schedule(instructions):
    """Static register dependencies, including quad-reduction FMNMX merges.

    PCs describe emitted order, not dynamic timings. Loop selection still needs
    the backedge and caller's inspection of masked versus unmasked iterations.
    """
    depths = {}
    maximum = 0
    positions = {key: [] for key in ('max', 'exp', 'wait0')}
    for address, op, operands in instructions:
        registers = re.findall(r'\bR\d+\b', operands)
        destination = re.match(r'R\d+\b', operands)
        if op.startswith('FMNMX'):
            depth = 1 + max((depths.get(reg, 0) for reg in registers[1:]), default=0)
            depths[registers[0]] = depth
            maximum = max(maximum, depth)
            positions['max'].append(hex(address))
        elif destination:
            depths[destination[0]] = depths.get(registers[1], 0) if op == 'MOV' and len(registers) > 1 else 0
        if op == 'MUFU.EX2':
            positions['exp'].append(hex(address))
        if op == 'WARPGROUP.DEPBAR.LE' and re.search(r'0x0\b', operands):
            positions['wait0'].append(hex(address))
    return {'fmnmx_dependency_depth': maximum, **{key + '_pcs': values if len(values) < 3 else [values[0], values[-1]] for key, values in positions.items()}}


def register_role(op, operands):
    if 'DEALLOC' in op or op.endswith('.DEC'):
        return 'producer'
    if 'ALLOC' in op or op.endswith('.INC'):
        return 'consumer'
    if re.search(r'0x18\b|\b24\b', operands):
        return 'producer'
    if re.search(r'0xf0\b|\b240\b', operands):
        return 'consumer'
    return 'register-region'


def report(text):
    print('Static SASS counts; SETMAX register instructions split linear address ranges.')
    print('Every consecutive WARPGROUP.ARRIVE pair in the consumer range is reported; inspect branch targets to identify steady-state versus prologue/epilogue.')
    for function in re.split(r'Function : ', text)[1:]:
        name, _, body = function.partition('\n')
        ins = []
        for line in body.splitlines():
            match = re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?[UP]\d+\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?);', line)
            if match:
                ins.append((int(match[1], 16), match[2], match[3]))
        if not any('SETMAX' in op for _, op, _ in ins):
            continue
        print('\nFUNCTION ' + name)
        boundaries = [(i, register_role(op, operands)) for i, (_, op, operands) in enumerate(ins) if 'SETMAX' in op]
        print('entry-before-register-split', counts(ins[:boundaries[0][0]]))
        for j, (begin, role) in enumerate(boundaries):
            end = boundaries[j+1][0] if j+1 < len(boundaries) else len(ins)
            region = ins[begin:end]
            print(f'{role} 0x{region[0][0]:x}..0x{region[-1][0]:x} {region[0][1:]} {counts(region)}')
            for addr, op, operands in region:
                if op.startswith(('LDL', 'STL')):
                    print(f'  local 0x{addr:x} {op} {operands}')
            if role == 'consumer':
                arrivals = [i for i in range(begin, end) if ins[i][1] == 'WARPGROUP.ARRIVE']
                for a, b in zip(arrivals, arrivals[1:]):  # noqa: RUF007 (login Python 3.6)
                    print(f'  ARRIVE-window 0x{ins[a][0]:x}..0x{ins[b][0]:x} {counts(ins[a:b])}')
        # Preserve backwards branches as evidence for loop selection.
        for addr, op, operands in ins:
            if op in ('BRA', 'BRA.U'):
                target = re.search(r'0x([0-9a-f]+)', operands)
                if target and int(target[1], 16) < addr:
                    print(f'  backedge 0x{addr:x} -> {operands}')
                    target_addr = int(target[1], 16)
                    body = [item for item in ins if target_addr <= item[0] <= addr]
                    census = counts(body)
                    if census['MUFU.EX2'] and census['F2FP']:
                        print(f'  arithmetic-backedge-loop 0x{target_addr:x}..0x{addr:x} {census}')
                        print(f'  softmax-schedule {softmax_schedule(body)}')


if __name__ == '__main__':
    report(Path(sys.argv[1]).read_text())
