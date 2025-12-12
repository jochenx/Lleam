# Lleam

`Lleam = LLM + Lean`

A verification tool that checks LLM statements for truth. Grounding is assured by fact extraction with a derivation and conclusion anchored in a formal proof. 
A theorem prover complements the LLM in arriving at a proof of statement, with opportunities to fix the proof if there are problems with it.

A set of steps with impartial LLM-as-judges will verify whether the proof is an acceptable answer to the original problem statement. While there is still reliance in the judgement of the LLM, which I consider to be somewhat unreliable, the main task of it is
comparing a translation of the proof into english against the original problem statement, and validating it. I consider that to be less prone to error than the original form of having an LLM reason without any sort of external validation. The Lean proof provides a grounding of the reasoning steps, but ultimately some judgement of an LLM is still
required. (The alternative would be to ask a person to check whether a translation of the proof is acceptable as an explanation of the original problem). 

The current implementation is still a bare-bones query and translation loop with an LLM and Lean. This will be build out over time.
There are currently only calls to Claude models, a next step would be to use a different LLM for translation and judging. 

Assumptions:

Lean 4 tools are installed, as well as Mathlib.
See https://github.com/leanprover for starting instructions.
