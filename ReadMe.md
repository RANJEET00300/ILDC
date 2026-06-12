---

# Architecture Summary: Iterative Latent Diffusion for Continuous KV Compression (ILDC)

---

## The Idea:
   Shift KV Cache compression from *discrete token pruning* (dropping words) to *generative latent representation*. 
Instead of a massive $N$-length context window, use a Diffusion block to iteratively denoise a dynamic set of $K$ tokens (where $K = N / X\\_factor$) conditioned on the $N$ context tokens. The result is a hyper-dense, continuous "thought vector" that perfectly represents the semantic structure of the dropped context, yielding massive memory efficiency.


## The Final Architecture Blueprint (How it works)

1.  **The Active Process:** The pre-trained LLM processes incoming text using a standard Attention (e.g., 8k tokens).
2.  **The Background Compression:** The Diffusion Model(Self-attention for latent-kvs and cross-attention with input LLM KVs) takes these $N$ dropped hidden states and iteratively compresses them down into $K$ continuous tokens (where $K = N / X\\_factor$). 
3.  **The Generation with COmpressed KVS:** Now the LLM can use the $K$ compressed KVs to predict the next token.
4. **Temporal Anchoring Strided RoPE:** The LLM can understand the timeline of these $K$ compressed KVs we got from the DM compressor as during diffusion, we are anchoring the latent-kvs with the stride rope embedding.
---



## **Training**: Consider the teacher-student like method:

What we can do is we can

1.  pass the whole 8K input + 2k output token to the base  model.
2.  keep the KVs for the 8K input and output vector for the 2K output and save it.

then as for the DM

3.  take the KVs condition for the DM compressor and output the compressed KVs....

4.  use the compressed KVs to predict the 2K output we want from it..

5.  use the saved 2K output vector from baseline model and estimate the loss for
    the predicted tokens conditioned on compressed tokens..

and then backprop....

**End-to-end Training**: Could be Implemented(Not yet implemented: we are training the compressor models attention blocks currently, but we can train the whole model together if needed).

**Result:** An LLM with very-long-context capability.



## Current Training Setup and Model:

* **AR_Model:** Gemma-3-1B-it ( Teacher Model|Auto Regressive Model)
* **DC_Model:** It is using gemma-3 trained MLP weight and new Cross Attention and Self Attention block weights are inititalized for training ( Latent Diffusion Compressor Model )

* **ILDC Model:** It uses the AR_Model and DC_Model together to compress the KVs of AR_Model and generate the next token using the compressed KVs.

* **Dataset:** Any Long context dataset: Havn't decided which one yet.

* **Training Flow:** Using 2K context tokens and 512 target tokens with two diffusion steps. During training we freeze the AR_Model weights and DC_Model-MLP weights and only train its attention blocks.

---

### What we are trying to do?
We are trying to answer the following questions:

1.  **Memory Compression Ratio:** "If we could successfully compress 100,000 tokens of KV cache into $K$ continuous latents based on our chosen $X\\_factor$."
2.  **Fidelity (Accuracy):** "If we could maintain 98%+ accuracy on long context tasks compared to uncompressed baselines."
3.  **Compute Latency:** "If we utilize ILDC-diffusion compressor model running asynchronously, how much it would improve the generation latency in long context tasks."


---
### Future Possibilities:
If we understand the working of the Iterative Latent Diffusion for KV Compressions: we could have a possibility of building *Infinite Context Model* with layered compression with different profile for compression like O(sqrt(n)), O(1) or O(log(n)) profile for context compression without losing much of the context fidelity. Here are some future implementations we are looking ahead:

1. **Hierarchical & Layered Compression**: With different compression profile for different layers upon layers.
    * Lower layers (close to input) can use higher compression ratios (e.g., O(sqrt(n)) or O(log(n)))
    * Higher layers (close to output) can use lower compression ratios (e.g., O(1))
2. **Progressive Refinement**: Multiple passes of compression with increasing fidelity.
3. **Adaptive Compression**: Compression ratio adjusts dynamically based on context complexity. 


**Best Use Case:**
    * **Continuous Learning & Evolving AI Systems:** Models can continuously learn from their environment and adapt their behavior over time with the compression model's self reflection on the long term layered context and use that memory to make better decisions in the future...
    * **Robotic AI Embodiement:** Live Models with high frequecy sensor data and vision + audio + text inputs for real time decision making and actions...
    * **Real-time Multimodal AI Assistant:** Live Models with vision + audio + text inputs for real time assistance...
    
   

## **Note:** This is a research project. The code and training setup are experimental and subject to change.

## Contributers are welcome!!!!