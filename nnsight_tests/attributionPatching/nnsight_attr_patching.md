
I plan to investigate the profiling logs more when I get the time. But currently, I just gave the entire log to gpt for it to summarize. 

Here's a summary of the profiling results for each computation stage, focusing on memory consumption and time taken per token and per prompt.

### Summary of Computation Stages

1. **trace_model**
    - **Memory Consumption:**
      - First run: 0.6113 MB per token, 8.5586 MB per prompt
      - Second run: 0.0040 MB per token, 0.0566 MB per prompt
      - Third run: 0.0755 MB per token, 1.0566 MB per prompt
      - Fourth run: 0.0151 MB per token, 0.2109 MB per prompt
    - **Time Taken:**
      - First run: 0.0274 seconds per token, 0.3839 seconds per prompt
      - Second run: 0.0334 seconds per token, 0.4682 seconds per prompt
      - Third run: 0.0325 seconds per token, 0.4549 seconds per prompt
      - Fourth run: 0.0326 seconds per token, 0.4559 seconds per prompt

2. **get_logit_diff**
    - **Memory Consumption:**
      - First run: -12.6562 MB per token, -12.6562 MB per prompt
      - Second run: 1.1250 MB per token, 1.1250 MB per prompt
      - Third run: -163.1406 MB per token, -163.1406 MB per prompt
      - Fourth run: 1.1562 MB per token, 1.1562 MB per prompt
      - Fifth run: 0.0938 MB per token, 0.0938 MB per prompt
      - Sixth run: 0.0000 MB per token, 0.0000 MB per prompt
      - Seventh run: 33.9688 MB per token, 33.9688 MB per prompt
      - Eighth run: 1.1094 MB per token, 1.1094 MB per prompt
    - **Time Taken:**
      - First run: 11.0086 seconds per token, 11.0086 seconds per prompt
      - Second run: 10.9645 seconds per token, 10.9645 seconds per prompt
      - Third run: 11.1175 seconds per token, 11.1175 seconds per prompt
      - Fourth run: 10.7600 seconds per token, 10.7600 seconds per prompt
      - Fifth run: 4.7118 seconds per token, 4.7118 seconds per prompt
      - Sixth run: 4.9168 seconds per token, 4.9168 seconds per prompt
      - Seventh run: 5.1682 seconds per token, 5.1682 seconds per prompt
      - Eighth run: 4.8547 seconds per token, 4.8547 seconds per prompt

3. **compute_baselines**
    - **Memory Consumption:**
      - 44.6719 MB per token, 44.6719 MB per prompt
    - **Time Taken:**
      - 60.5733 seconds per token, 60.5733 seconds per prompt

4. **ioi_metric**
    - **Memory Consumption:**
      - First run: 0.1562 MB per token, 0.1562 MB per prompt
      - Second run: 0.0000 MB per token, 0.0000 MB per prompt
      - Third run: 1.2969 MB per token, 1.2969 MB per prompt
    - **Time Taken:**
      - First run: 23.6270 seconds per token, 23.6270 seconds per prompt
      - Second run: 23.4330 seconds per token, 23.4330 seconds per prompt
      - Third run: 11.4332 seconds per token, 11.4332 seconds per prompt
      - Fourth run: 11.8817 seconds per token, 11.8817 seconds per prompt

5. **trace_attention**
    - **Memory Consumption:**
      - 187.0312 MB per token, 187.0312 MB per prompt
    - **Time Taken:**
      - 28.1280 seconds per token, 28.1280 seconds per prompt

6. **patch_attention_heads**
    - **Memory Consumption:**
      - 88.1875 MB per token, 88.1875 MB per prompt
    - **Time Taken:**
      - 2.9072 seconds per token, 2.9072 seconds per prompt

7. **patch_positions**
    - **Memory Consumption:**
      - -21.7812 MB per token, -21.7812 MB per prompt
    - **Time Taken:**
      - 2.3608 seconds per token, 2.3608 seconds per prompt

### Overall Estimates for Experiments at Scale

- **Compute Commitment:**
  - Given the variability in memory usage across different stages, careful memory management and potentially batching of operations will be critical.
  - Large memory usage in stages like `trace_attention` suggests that scaling up might require hardware with significant memory capacity.

- **Time Commitment:**
  - Time per token ranges from around 0.0274 seconds to 11.1175 seconds across different stages.
  - Time per prompt ranges from around 0.3839 seconds to 60.5733 seconds across different stages.
  - The `trace_model` and `get_logit_diff` functions, being repeatedly called, will dominate the runtime.
  - `trace_attention` and `ioi_metric` also contribute significantly to the time.
