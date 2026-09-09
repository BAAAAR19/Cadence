| KV core          | Component             | Inclusive   | Self   |
|:-----------------|:----------------------|:------------|:-------|
| Python reference | scheduler             | 100.00%     | 0.00%  |
| Python reference | model runner          | 100.00%     | 92.57% |
| Python reference | tokenizer (llama.cpp) | 4.20%       | 4.20%  |
| Python reference | sampling (numpy)      | 3.23%       | 3.23%  |
| Python reference | kv: radix cache       | 0.02%       | 0.00%  |
| C++17 core       | scheduler             | 100.00%     | 0.00%  |
| C++17 core       | model runner          | 100.00%     | 92.73% |
| C++17 core       | tokenizer (llama.cpp) | 4.06%       | 4.06%  |
| C++17 core       | sampling (numpy)      | 3.21%       | 3.21%  |

* **Python reference**: 59,277 samples at 381 Hz over 155 s, of which the scheduler thread was busy 99%. 2,180 engine steps, mean 70.3 ms. 324 completed requests at 1.9 rps.
* **C++17 core**: 59,932 samples at 387 Hz over 155 s, of which the scheduler thread was busy 98%. 2,671 engine steps, mean 56.8 ms. 324 completed requests at 1.9 rps.
