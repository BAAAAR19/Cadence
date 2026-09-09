"""Week 4: prompt + queue snapshot -> feature vector.

Deliberately empty until Week 4. The one design rule recorded now, because it
is the pitfall that invalidates the whole experiment: the extractor may read
only the ``Request`` and a snapshot of scheduler state taken *at admission
time*. It must never see anything derived from how the request actually turned
out -- not its output length, not its measured latency. That leakage produces a
suspiciously good predictor and a coverage guarantee that means nothing.
"""
