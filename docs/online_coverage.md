| Arm | Nominal | Empirical | admitted & completed | violations | mean bound (s) | mean actual (s) |
|:--|--:|--:|--:|--:|--:|--:|
| 5  + conformal admission (95%) | 95% | **100.0%** | 2876 | 0 | 2.60 | 0.72 |
| 5  + conformal admission (80%) | 80% | **92.2%** | 4091 | 321 | 2.22 | 1.14 |

Coverage over every request the controller admitted and saw finish, across three seeds and six offered loads, from the traces it wrote while it was deciding. This is deliberately *not* the held-out coverage in the Week 4 fit: that one is a check on the method, and this one is a check on the deployment. The finite-sample guarantee does not apply here at all -- the calibration set was collected with admission off, and the controller changes which requests run, so the exchangeability the theorem needs is gone by construction. The gap between the mean bound and the mean actual latency is the price of a bound that is valid rather than sharp.
