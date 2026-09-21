# API inference: exploratory results, 2026-09-21

Model: `qwen/qwen3.6-35b-a3b`, reasoning requested off, temperature=0.
24/24 measured requests and 6/6 warmups succeeded; successful usage coverage 100%.
Provider reported 44,220 prompt and 1,442 completion tokens including warmups.
Cost units are not established here; no currency claim is made.

| Prompt tokens | Concurrency | Median first text, s | Median complete, s | Successful req/s | Reported completion tok/s |
| --- | --- | --- | --- | --- | --- |
| 201 | 1 | 1.173 | 1.884 | 0.509 | 24.71 |
| 201 | 2 | 1.292 | 1.789 | 1.115 | 54.10 |
| 201 | 4 | 1.215 | 2.112 | 1.728 | 86.38 |
| 2747 | 1 | 1.224 | 1.722 | 0.558 | 26.39 |
| 2747 | 2 | 1.139 | 1.760 | 0.971 | 44.90 |
| 2747 | 4 | 1.330 | 1.720 | 1.993 | 95.64 |

Four observations per cell, short generated answers, warm shared connection pool and repeated
prompts: this is a smoke benchmark, not a capacity estimate. Larger context did not consistently
increase latency in this run; caching and provider noise were not controlled. Concurrency 4
increased measured throughput over concurrency 1; this does not establish an optimal limit.
No p95, exact token timing, GPU measurements or production configuration changes.
Raw samples and provenance: `artifacts/inference_api_20260921_live/report.json` (ignored).
Initial sandbox run was interrupted without a completed cell; its directory is not a result.
Network run outside sandbox completed. 26 benchmark/client tests passed; scoped Ruff and mypy passed.
See README.md for reproduction and measurement definitions.
