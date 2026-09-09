**Longest-prefix match, by prompt length**

|   Prompt tokens |   Python (us) |   C++ (us) |   of which pybind11 marshalling | Speedup   |
|----------------:|--------------:|-----------:|--------------------------------:|:----------|
|             128 |          4.55 |       2.69 |                            2.04 | 1.7x      |
|             256 |          8.96 |       4.87 |                            4.52 | 1.8x      |
|             512 |         18.81 |       8.8  |                            9.16 | 2.1x      |
|            1024 |         39.48 |      19.1  |                           19.6  | 2.1x      |
|            2048 |         86.61 |      37.02 |                           33.79 | 2.3x      |

**Allocator operations**

| Operation                                |   Python (us) |   C++ (us) | Speedup   |
|:-----------------------------------------|--------------:|-----------:|:----------|
| can_append (once per sequence per token) |         0.246 |      0.16  | 1.5x      |
| alloc + release of a 34-block table      |         4.175 |      1.272 | 3.3x      |

**What that is as a share of one engine step** (mean step 70.3 ms, measured)

|   Running batch | Python   | C++     | Python, share of a step   | C++, share of a step   |
|----------------:|:---------|:--------|:--------------------------|:-----------------------|
|               8 | 22.7 us  | 11.4 us | 0.032%                    | 0.016%                 |
|              12 | 24.7 us  | 12.6 us | 0.035%                    | 0.018%                 |
|              24 | 30.6 us  | 16.5 us | 0.044%                    | 0.023%                 |

**The price of releasing the GIL, which is why the port does not**

| Measurement                                  |   ns per call |
|:---------------------------------------------|--------------:|
| a bound no-op, GIL held throughout           |            48 |
| the same no-op, GIL released and re-acquired |            83 |
| cost of the release/re-acquire pair          |            35 |
