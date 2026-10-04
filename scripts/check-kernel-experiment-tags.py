"""Validate explicit bounded kernel-build/probe tags before any expensive work."""
import os


def validate(message):
    build_only = '[build-only]' in message
    probe_tags = ['[cutlass-grid-probe]', '[moe-routing-probe]']
    probe = any(tag in message for tag in probe_tags)
    if all(tag in message for tag in probe_tags):
        raise ValueError('distinct finite probes require separate tracked commits')
    if not (build_only or probe):
        return
    if build_only and probe:
        raise ValueError('build-only and finite probes are separate tracked stages')
    if '[cluster:aspire2ap]' not in message or any(tag in message for tag in ['[cluster:genkai]', '[cluster:firmus]']):
        raise ValueError('kernel build/probe requires only cluster:aspire2ap')
    if '[profile]' in message:
        raise ValueError('finite kernel stages do not capture a profiler')
    if build_only and not message.startswith('[require-build]'):
        raise ValueError('build-only requires the require-build prefix')
    if probe and ('[nodes:1]' not in message or any(tag in message for tag in ['[nodes:2]', '[nodes:4]']) or message.startswith('[require-build]')):
        raise ValueError('kernel probe requires nodes:1 and a separately built reusable image')


if __name__ == '__main__':
    validate(os.environ.get('QWEN_COMMIT_MESSAGE', ''))
