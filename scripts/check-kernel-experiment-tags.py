"""Validate explicit bounded kernel-build tags before any expensive work."""
import os


def validate(message):
    if '[kernel-bench]' in message:
        # Each bench reuses its pinned image on one node. Multiple explicitly
        # selected supported clusters have independent comparisons and results.
        clusters = [tag for tag in ['[cluster:aspire2ap]', '[cluster:firmus]'] if tag in message]
        if not clusters or any(tag in message for tag in ['[cluster:genkai]', '[dynamo]', '[profile]', '[build-only]']):
            raise ValueError('kernel bench requires cluster:aspire2ap or cluster:firmus')
        if '[nodes:1]' not in message or any(tag in message for tag in ['[nodes:2]', '[nodes:4]', '[require-build]']):
            raise ValueError('kernel bench requires nodes:1 and the pinned image')
    if '[build-only]' not in message:
        return
    clusters = [tag for tag in ['[cluster:aspire2ap]', '[cluster:firmus]'] if tag in message]
    if len(clusters) != 1 or any(tag in message for tag in ['[cluster:genkai]', '[dynamo]']):
        raise ValueError('kernel build requires exactly one of cluster:aspire2ap or cluster:firmus, without dynamo')
    if '[profile]' in message:
        raise ValueError('build-only stages do not capture a profiler')
    if not message.startswith('[require-build]'):
        raise ValueError('build-only requires the require-build prefix')


if __name__ == '__main__':
    validate(os.environ.get('QWEN_COMMIT_MESSAGE', ''))
