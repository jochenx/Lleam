# Lleam

`Lleam = LLM + Lean`

A verification tool to help an LLM make statements about truth. Grounding is facilitated by feeding in facts, derivation and conclusion, all anchored in a formal proof. 

Here, the LLM writes a Lean program, which the theorem prover compiles and accepts or rejects. This loop repeats until an acceptable proof is found.

There is still a formalization gap i.e. does the proof cover an original problem statement, and are the assumptions likely to be valid and appropriate.
While there is still reliance in the judgement of the LLM, which I consider to be somewhat unreliable, the main task of the judge is
comparing a translation of the proof into english against the original problem statement, and validating it. I consider that to be less prone to error than the original form of having an LLM reason without any sort of external validation. The Lean proof provides a grounding of the reasoning steps, but ultimately some judgement of an LLM is still required. (The alternative would be to ask a human to check whether a translation of the proof is acceptable as an explanation of the original problem).

The current implementation is still a bare-bones query and translation loop with an LLM and Lean. This will be build out over time.
There are currently only calls to Claude models, a next step would be to use a different LLM for translation and judging. 

Assumptions:

Lean 4 tools are installed, as well as Mathlib.
See https://github.com/leanprover for starting instructions.
