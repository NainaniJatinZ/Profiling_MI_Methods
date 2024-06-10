# Profiling_MI_Methods

I am a researcher facing a lot of issue with memory when doing Mech Interp at scale. This library is made to memory and time profile different methods and libraries used in Mech Interp. Primary model will probably be gpt2-small for the beginning. Planning to use llama3-8b for larger experiments. 

# TODOS:

Libraries: 
- [ ] nnsight
  - [ ] Activation Patching
  - [ ] Attribution Patching
  - [ ] SAE

- [ ] transformerLens
  - [ ] Activation Patching
  - [ ] Attribution Patching
  - [ ] SAE

- [ ] EluetherAI
  - [ ] SAE 

- [ ] will add more as suggested

Hardware:
- [ ] M3 Pro
- [ ] a100
- [ ] rtx8000
- [ ] titanx
- [ ] cpu only

# Observations:

This repo will also help me understand the inner workings of the methods and how various researchers are optimizing their experiments. I plan to put out all my findings in this repo. Hopefully it benefits someone else too! 


- [NNsight Attribution Patching Tutorial Profiling (CPU only)](nnsight_tests/attributionPatching/nnsight_attr_patching.md)


## What should we care about? 

For each computation stage, I care about:

- Memory consumption per token, per prompt 

- Time taken per token, per prompt 

Overall: 

- Estimates of the compute and time commitment needed for experiments at scale