"""Validate explicit bounded kernel-build tags before any expensive work."""
import os


def validate(message):
    if '[build-only]' not in message:
        return
    if '[cluster:aspire2ap]' not in message or any(tag in message for tag in ['[cluster:genkai]', '[cluster:firmus]']):
        raise ValueError('kernel build requires only cluster:aspire2ap')
    if '[profile]' in message:
        raise ValueError('build-only stages do not capture a profiler')
    if not message.startswith('[require-build]'):
        raise ValueError('build-only requires the require-build prefix')


if __name__ == '__main__':
    validate(os.environ.get('QWEN_COMMIT_MESSAGE', ''))
