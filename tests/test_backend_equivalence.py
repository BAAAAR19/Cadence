"""Step 1.1's exit test, and the foundation of everything after it.

If decoding two sequences in one batched ``decode_step`` does not produce what
decoding them separately produces, then every throughput number obtained by
batching is measuring a different computation than the baseline, and the
comparison is meaningless.

The assertion is *top-token agreement over a corpus*, not bit-exact logits.
Batching genuinely changes numerics -- two sequences decoded together hit
different reduction orders in the matmul than the same sequences decoded apart
-- so a bit-exactness test fails confusingly and invites you to "fix" a non-bug.
"""

from __future__ import annotations

import pytest

from cadence.engine.backends.base import SeqState

PROMPTS = [
    "List three colours.",
    "What is the capital of Japan?",
    "Explain gravity in one sentence.",
    "Write a haiku about rain.",
    "Name two prime numbers greater than ten.",
    "What does HTTP stand for?",
]
N_STEPS = 24


def _generate(runner, prompt_ids: list[int], seq_id: int, n: int) -> list[int]:
    seq = SeqState(seq_id=seq_id, prompt_ids=list(prompt_ids))
    out = [runner.prefill([seq])[0]]
    for _ in range(n - 1):
        if seq.finished:
            break
        out.append(runner.decode_step([seq])[0])
    runner.free(seq_id)
    return out


@pytest.mark.slow
def test_batched_decode_matches_separate_decode(llama_runner):
    runner = llama_runner
    runner.reset()
    ids = [runner.tokenize(runner.format_chat([{"role": "user", "content": p}])) for p in PROMPTS]

    # Separately, one sequence at a time.
    alone = [_generate(runner, p, i, N_STEPS) for i, p in enumerate(ids)]
    runner.reset()

    # Together, in one batch, advanced a token at a time.
    seqs = [SeqState(seq_id=i, prompt_ids=list(p)) for i, p in enumerate(ids)]
    together: list[list[int]] = [[t] for t in runner.prefill(seqs)]
    for _ in range(N_STEPS - 1):
        live = [s for s in seqs if not s.finished]
        if not live:
            break
        toks = runner.decode_step(live)
        for s, tok in zip(live, toks, strict=True):
            together[seqs.index(s)].append(tok)
    for i in range(len(ids)):
        runner.free(i)

    agree = total = 0
    for a, b in zip(alone, together, strict=True):
        for x, y in zip(a, b, strict=False):
            total += 1
            agree += x == y
    rate = agree / total
    assert rate >= 0.99, f"top-token agreement {rate:.3f} over {total} positions"


@pytest.mark.slow
def test_prefill_chunking_is_equivalent_to_one_shot(llama_runner):
    """Chunked prefill must not change what the model produces -- only when it
    produces it."""
    runner = llama_runner
    runner.reset()
    prompt = runner.tokenize(
        runner.format_chat(
            [
                {"role": "system", "content": "You are terse. " * 200},
                {"role": "user", "content": "Say hello."},
            ]
        )
    )
    assert len(prompt) > 512, "prompt must exceed one chunk for this to test anything"

    one_shot = _generate(runner, prompt, 0, 12)
    runner.reset()

    seq = SeqState(seq_id=1, prompt_ids=list(prompt))
    tok = None
    n_chunks = 0
    while tok is None:
        tok = runner.prefill_chunk([seq], budget=128)[0]
        n_chunks += 1
    chunked = [tok]
    for _ in range(11):
        if seq.finished:
            break
        chunked.append(runner.decode_step([seq])[0])
    runner.free(1)

    assert n_chunks > 1, "the prompt was not actually chunked"
    agree = sum(x == y for x, y in zip(one_shot, chunked, strict=False))
    assert agree / min(len(one_shot), len(chunked)) >= 0.99


def test_mock_runner_batches_like_it_steps(cfg_mock):
    """The same property, on the deterministic backend, so CI checks the
    contract even without a model on disk."""
    from cadence.engine.backends.mock import MockRunner

    runner = MockRunner(cfg_mock)
    ids = [runner.tokenize(p) for p in PROMPTS]

    alone = [_generate(runner, p, i, 8) for i, p in enumerate(ids)]
    seqs = [SeqState(seq_id=i, prompt_ids=list(p)) for i, p in enumerate(ids)]
    together = [[t] for t in runner.prefill(seqs)]
    for _ in range(7):
        toks = runner.decode_step(seqs)
        for i, tok in enumerate(toks):
            together[i].append(tok)
    assert alone == together
