# VeriGround
A verification tool that checks LLM statements for truth. Grounding is assured by fact extraction with a derivation and conclusion anchored in a formal proof. A theorem prover complements the LLM in arriving at a proof of statement by a dynamic proposition and correction loop.

The initial implementation is a Q&A module as front-end, a fact extraction engine, a LLM that proposes proofs and Lean 4 interface to run the proof. Feedback is sent to the LLM in the case of problems with the proof. Answers ultimately supplied by the LLM must be tightly coupled to the proof.
