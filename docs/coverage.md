|    α |   Nominal (1−α) |   Empirical, offline | 95% CI           |   Q (log-ratio) |   Mean U (s) | Would admit   |
|-----:|----------------:|---------------------:|:-----------------|----------------:|-------------:|:--------------|
| 0.2  |            0.8  |               0.8503 | [0.8225, 0.8744] |          -0.005 |         9.9  | 19.9%         |
| 0.1  |            0.9  |               0.9299 | [0.9091, 0.9463] |           0.176 |        12.07 | 15.0%         |
| 0.05 |            0.95 |               0.9684 | [0.9530, 0.9789] |           0.355 |        14.63 | 10.6%         |
| 0.01 |            0.99 |               0.9973 | [0.9900, 0.9992] |           0.73  |        21.74 | 3.7%          |

And the same quantity measured while the controller was deciding — where the calibration set no longer describes what runs, because the controller chose it:

| Arm                            |   Nominal (1−α) |   Empirical, online |   Admitted and completed |   Censored (client gave up) |   Mean U (s) |   Mean realised E2E (s) |
|:-------------------------------|----------------:|--------------------:|-------------------------:|----------------------------:|-------------:|------------------------:|
| 5  + conformal admission (95%) |            0.95 |              0.9978 |                      894 |                           0 |         2.67 |                    0.78 |
| 5  + conformal admission (80%) |            0.8  |              0.876  |                     1387 |                           0 |         2.37 |                    1.37 |
