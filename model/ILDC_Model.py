import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


from .helper_fn import RMSNorm, apply_rotary_pos_emb, get_timestep_embedding
from .config import ModelConfig, TrainingConfig

from .AR_Model import CausalLM
from .DC_Model import CompDiTModel




# ===========================================================================================================

# THE ILDC based LLM ARCHITECTURE
"""
A Standard Auto-regressive model LLM; based on Gemma-3-1b
Modified to support ILDC (Iterative Latent Diffusion for Continuous KV Compression)
"""

# ===========================================================================================================



def load_models(checkpoint_dir, device):

    print(f"Using device: {device}")
    print("Loading ILDC Architecture...")

    checkpoint = checkpoint_dir
    
    model_id = "google/gemma-3-1b-it"
    # Instantiate models
    ar_model = CausalLM().to(torch.bfloat16)
    dc_model = CompDiTModel().to(torch.bfloat16)
    
    print("Downloading and mapping HF weights...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    gemma_state_dict = {}
    for key, value in hf_model.state_dict().items():
        new_key = key[6:] if key.startswith("model.") else key
        if "rotary_emb" not in new_key:
            gemma_state_dict[new_key] = value
    
    ar_model.load_state_dict(gemma_state_dict, strict=False)
        
    # We use ar_model mlp weight and freeze that, only train other weights
    if checkpoint is None:
        print("⚠️ Checkpoint not provided! Instantiating with untrained compressor weights")

        # Iterate over the ModuleLists to load the MLPs layer-by-layer
        for dc_layer, ar_layer in zip(dc_model.comp_layers, ar_model.gen_layers):
            dc_layer.mlp.load_state_dict(ar_layer.mlp.state_dict())

    elif os.path.exists(checkpoint): # Fixed from checkpoint_path
        print(f"Loading trained compressor weights from {checkpoint}...")
        state_dict = torch.load(checkpoint, map_location="cpu")
        dc_model.load_state_dict(state_dict)
        
    else:
        print(f"⚠️ Checkpoint '{checkpoint}' not found! Using untrained compressor weights for test.")

        # Iterate over the ModuleLists to load the MLPs layer-by-layer
        for dc_layer, ar_layer in zip(dc_model.comp_layers, ar_model.gen_layers):
            dc_layer.mlp.load_state_dict(ar_layer.mlp.state_dict())


    del hf_model

    print("Weights loaded successfully!")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    #     return ar_model.to("cuda:0"), dc_model.to("cuda:1")
    # else:
    return ar_model.to(device), dc_model.to(device)



class ILDC(nn.Module):
    def __init__(self, checkpoint_path=None, device="cpu"):
        super().__init__()

        self.config = Config()
        self.device = device

        # Load Baseline Gemma 3 1B
        model_id = "google/gemma-3-1b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Instantiate and Load weights
        self.ar_model, self.dc_model = load_models(checkpoint_path, device)
        
        # Freeze base LLM weights initially
        for param in self.ar_model.parameters():
            param.requires_grad = False

        # Freeze base mlp weights initially
        for dc_layer in self.dc_model.comp_layers:
            for param in dc_layer.mlp.parameters():
                param.requires_grad = False




    # ===================================================================================
    # Compressing the KV-cache with Diffusion at an timestep with given laten and KVs 
    # Latent KV Compression Model

    def kv_compress(self, latent_states, latent_kvs, kv_caches, start_step, diff_steps:int=1, start_pos=0):   
        """
        latent_states: torch.tensor of shape (B, K_latent, latent_size) [Diffusion Latent States]

        kv_caches: torch.Tensor of shape (
                                    B, num_hidden_layers, seq_len, 2, num_heads, head_dim
                                ) [KVs Compressed and Uncompressed Both]

        start_step: int [the timestep of input Latent-step if has been processed]
        diff_steps: int [Diffusion steps to go]
        start_pos: int [Start-position of uncompressed KVs in KV-caches which we are going to compress]

        Return: Latent States & New compressed KVs
        """

        B, num_layer, seq_len, _, num_heads, H = kv_caches.shape
        _, K_latent_len, latent_size = latent_states.shape 

        device = self.device
        target_dtype = self.config.dtype
      

        # 🔴 Ensure ALL incoming KV caches are strictly BFloat16
        safe_kv_caches = kv_caches.to(target_dtype)
        
        # The latent-states and compressed-kvs will be collect on each diffusion steps and will be given to AR model for student generation pass
        # Pre-allocate tensors Shape: (Steps, Batch, K_latent_len, Latent)
        all_latent_states = torch.empty(
            (diff_steps, B, K_latent_len, latent_size), 
            device=device, dtype=target_dtype
        )
        
        # Determine KV shapes (Adjust these based on your dc_model output)
        num_layers = self.config.num_hidden_layers
        
        all_compressed_kvs = torch.empty(
            (diff_steps, B, num_layers, K_latent_len, 2, num_heads, H),
            device=device, dtype=target_dtype
        )


        # 3. Diffusion Loop
        for t in range(diff_steps):
            timestep = t + start_step
            
            # Dit Model Pass
            # h: (B, K_latent_len, latent_size), kv: (B, layers, K_latent_len, 2, heads, head_dim)
            latent_states, latent_kvs = self.dc_model(latent_states, latent_kvs, safe_kv_caches, timestep, start_pos)
            
            # Direct assignment to pre-allocated memory
            all_latent_states[t] = latent_states
            all_compressed_kvs[t] = latent_kvs


        # 4. Reshape to (Batch, Steps, ...) 
        output_state = all_latent_states.transpose(0, 1).contiguous()

        output_compressed_kvs = all_compressed_kvs.transpose(0, 1).contiguous()

        return output_state, output_compressed_kvs




    # ================================================================================================
    # full-bptt forward....
    def bptt_forward(
        self, input_ids, context_ids=None, 
        active_kv_caches= None, active_compressed_kv= None, 
        latent_states= None,
        start_step = 0, diff_steps=1, start_pos=None
        ): 
        """
        Forward pass mainly used for Student Pass during training.
        input_ids: the preceeding token which will be in native embedding 
        context_ids: the preceeding token which will be compressed before feeding to AR
        """
        X_factor = self.config.X_factor
        target_dtype = self.config.dtype
        device = self.device

        # upcoming token for training
        B , future_len = input_ids.shape
        current_ids = input_ids

        # current preceeding token in context
        current_len = 0
        if context_ids is not None:
            _, current_len = context_ids.shape
            current_ids = torch.cat([context_ids, current_ids], dim=1)

       
        # effective seq-len to be compressed
        eff_seq_len = current_len

        if active_kv_caches is not None:
            _, _, KV_len, _, _, _ = active_compressed_kv.shape
            eff_seq_len += KV_len
            dropped_kvs =  active_kv_caches

        K_latent_len = (eff_seq_len + X_factor -1)// X_factor

        # Our start-pos for compression will start after the cmpressed kvs for efficient compression
        # if both the compressed_kv and start-pos are given,...we assume is as mistake and overwrite the start-pos
        if active_compressed_kv is not None:
            _, _, CompKV_len, _, _, _ = active_compressed_kv.shape
            if start_pos is not None:
                print(f"Overwriting start-pos with Comp_KV seq len: {CompKV_len}!")
            start_pos = CompKV_len
        
        if start_pos is None:
            start_pos = 0


        # ===================================== Teacher ==============================================================
         

        
        # 1. TEACHER PASS (No Gradients)
        with torch.no_grad():
            # Process the dropped context to populate its KV cache
            teacher_outputs = self.ar_model(
                input_ids = current_ids,   # combined input_ids + context_ids
                active_kv_caches=active_kv_caches,
                compressed_kv_caches=active_compressed_kv
            )

            # Teacher truth for the future tokens
            teacher_hidden_states = teacher_outputs["hidden_states"]
            teacher_output_logits = teacher_outputs["logits"]
            teacher_active_kv_caches = teacher_outputs["active_kv_caches"]

            # logit and hidden-states for future training
            h_teacher = teacher_hidden_states[:, -int(future_len):, :] # [B, S, H]
            logits_teacher = teacher_output_logits[:, -int(future_len):, :]       # [B, S, V]

            # total uncompressed kvs, from input kV_caches and the new uncompressed context KVs to be compressed
            # TODO: Needs to understand the ordering of Concatings of KVs all the way from begining...
            uncomp_context_kvs = teacher_active_kv_caches[:, :, :int(eff_seq_len), ... ]
    


        # ============================ KV Compression==============================================================
        # Totals KVs on which to conditioned for compression: Compressed_KVs and Uncompressed KVs too!
        if active_compressed_kv is not None:
            cond_context_kvs = torch.cat([active_compressed_kv, uncomp_context_kvs], dim=2)
        else: 
            cond_context_kvs = uncomp_context_kvs


        if latent_states is None:
            # if No latent_states, we assume, it is the first step and so we just put a noisy latent
            # and also make start_step = 0
            start_step = 0
            latent_states = torch.randn(
                B, 
                K_latent_len, 
                self.config.hidden_size, 
                device=self.device, 
                dtype=target_dtype
            )

        # Compress the extracted KVs using the DM tower 
        T_latent_states, T_newcompressed_kv_caches = self.kv_compress(latent_states, cond_context_kvs, start_step, diff_steps, start_pos)
            
        B, T, L, Comp_Seq, _, H, D = T_newcompressed_kv_caches.shape
        
        # Concat new and previous compressed KVs
        if active_compressed_kv is not None:
            T_active_compressed_kv = active_compressed_kv.unsqueeze(1).expand(B, T, L, CompKV_len, 2, H, D)
            T_compressed_kv_caches = torch.cat([T_active_compressed_kv, T_newcompressed_kv_caches], dim=3)
        else:
            T_compressed_kv_caches = T_newcompressed_kv_caches

        # =============================== Student ===========================================================
        
        # Process the target tokens utilizing the compressed KVs for historical context
        # As the DM_Tower goes with diffusion steps, and we are conditions the LLM 
        # on each diffusion step, we flatten then along step dim and pass the same input_ids
        # T times with each step having its own compressed_kv_caches for different steps of same forward pass.

        # Expand and Flatten input_ids: [B, S] -> [B, T, S] -> [B*T, S]
        _, S = input_ids.shape
        input_ids_flat = input_ids.unsqueeze(1).expand(B, T, S).reshape(B * T, S)

        _, _, _, Comp_L, _, _, _   = T_compressed_kv_caches.shape
        
        # Flatten KV caches: [B, T, L, S, 2, H, D] -> [B*T, L, S, 2, H, D]
        compressed_kv_flat = T_compressed_kv_caches.reshape(B * T, L, Comp_L, 2, H, D)

        
        student_outputs = self.ar_model(
            input_ids = input_ids_flat, 
            active_kv_caches=None,  # As we have compressed all the KV caches
            compressed_kv_caches=compressed_kv_flat
        )
        
        # Shapes: [B*T, S, V] and [B*T, S, H]
        student_logits = student_outputs["logits"] 
        h_student = student_outputs["hidden_states"]

        _, S, V = student_logits.shape
        _, _, H = h_student.shape

        # Shapes: [B, T, S, V] and [B, T, S, H]
        student_logits = student_logits.reshape(B, T, S, V)
        student_h = h_student.reshape(B, T, S, H)

        # ============================= Output =============================================================
        
        train_output = {
            "teacher": {
                "hidden": h_teacher,  #(B, S, V)
                "logits": logits_teacher
            },
            "student":{
                "T_hidden": student_h, #(B, T, S, V)
                "T_logits": student_logits 
            }
        }


        return train_output, T_latent_states




    # Implementation with memory savings: trading with compute:
    # =================================================================================================
    # Sifting from BPTT to single step block training implementation...
    def forward(
        self, input_ids=None, context_ids=None, active_kv_caches= None, active_compressed_kv= None, start_pos=0,
        start_step = 0, diff_steps=2, 
        ): 
        """
        diff_steps: T
        """
        device = self.device
        target_dtype = self.config.dtype


        X_factor = self.config.X_factor
        target_dtype = self.config.dtype
        device = self.device

        # upcoming token for training
        B , future_len = input_ids.shape
        current_ids = input_ids

        # current preceeding token in context
        current_len = 0
        if context_ids is not None:
            _, current_len = context_ids.shape
            current_ids = torch.cat([context_ids, current_ids], dim=1)

       
        # effective seq-len to be compressed
        eff_seq_len = current_len

        if active_kv_caches is not None:
            _, _, KV_len, _, _, _ = active_compressed_kv.shape
            eff_seq_len += KV_len
            dropped_kvs =  active_kv_caches

        K_latent_len = (eff_seq_len + X_factor -1)// X_factor

        # ===================================== Teacher ==============================================================
         

        
        # 1. TEACHER PASS (No Gradients)
        with torch.no_grad():
            # Process the dropped context to populate its KV cache
            teacher_outputs = self.ar_model(
                input_ids = current_ids,   # combined input_ids + context_ids
                active_kv_caches=active_kv_caches,
                compressed_kv_caches=active_compressed_kv
            )

            # Teacher truth for the future tokens
            teacher_hidden_states = teacher_outputs["hidden_states"]
            teacher_output_logits = teacher_outputs["logits"]
            teacher_active_kv_caches = teacher_outputs["active_kv_caches"]

            # logit and hidden-states for future training
            h_teacher = teacher_hidden_states[:, -int(future_len):, :] # [B, S, H]
            logits_teacher = teacher_output_logits[:, -int(future_len):, :]       # [B, S, V]

            # total uncompressed kvs, from input kV_caches and the new uncompressed context KVs to be compressed
            uncomp_context_kvs = teacher_active_kv_caches[:, :, :int(eff_seq_len), ... ]



        # ============================ KV Compression==============================================================

        # Our start-pos for compression will start after the cmpressed kvs for efficient compression
        # if both the compressed_kv and start-pos are given,...we assume is as mistake and overwrite the start-pos

        if active_compressed_kv is not None:
            _, _, CompKV_len, _, _, _ = active_compressed_kv.shape
            if start_pos is not None:
                print(f"Overwriting start-pos with Comp_KV seq len: {CompKV_len}!")
            start_pos = CompKV_len
            # Totals KVs on which to conditioned for compression: Compressed_KVs and Uncompressed KVs too!
            cond_context_kvs = torch.cat([active_compressed_kv, uncomp_context_kvs], dim=2)
        else: 
            cond_context_kvs = uncomp_context_kvs


        current_ground_latent_states = torch.randn(
            B, 
            K_latent_len, 
            self.config.hidden_size, 
            device=self.device, 
            dtype=target_dtype
        )

        latent_kvs = None
        
        # Compress the extracted KVs using the DM tower 
        # we assume that during training, the previous steps has been learned before start-step which can give us ground latent values..
        if start_step > 0:
             # it is the first step and so we just put a noisy latent
            with torch.no_grad():
                T_latent_states, T_compressed_kv_caches = self.kv_compress(
                    current_ground_latent_states, 
                    latent_kvs,
                    cond_context_kvs,
                    0, # back_step. ground -> c_ground | diff_steps. c_ground -> current_train_step...
                    start_step, # all the way from start to current...
                    start_pos)

                current_ground_latent_states = T_latent_states[:, -1, ...]
                latent_kvs = T_compressed_kv_caches[:, -1, ...]




        # forward training
        T_latent_states, T_newcompressed_kv_caches = self.kv_compress(
            current_ground_latent_states,
            latent_kvs, 
            cond_context_kvs, 
            start_step, # diff_steps. c_ground -> current_train_step...
            diff_steps,        
            start_pos)

            

            
        B, T, L, Comp_Seq, _, H, D = T_newcompressed_kv_caches.shape
        newcompressed_kv_caches = T_newcompressed_kv_caches[:, -1, ...]
        
        # Concat new and previous compressed KVs
        if active_compressed_kv is not None:
            compressed_kv_caches = torch.cat([active_compressed_kv, newcompressed_kv_caches], dim=2)
        else:
            compressed_kv_caches = newcompressed_kv_caches

        # =============================== Student ===========================================================

        
        student_outputs = self.ar_model(
            input_ids = input_ids, 
            active_kv_caches=None,  # As we have compressed all the KV caches
            compressed_kv_caches=compressed_kv_caches
        )
        
        # Shapes: [B, S, V] and [B, S, H]
        logits_student = student_outputs["logits"] 
        h_student = student_outputs["hidden_states"]


        # ============================= Output =============================================================
        
        train_output = {
            "teacher": {
                "hidden": h_teacher,  #(B, S, V)
                "logits": logits_teacher
            },
            "student":{
                "hidden": h_student, #(B, S, V)
                "logits": logits_student 
            }
        }


        return train_output, T_latent_states[:, -1, ...].detach()



    # ===================================================================================
    # Auto-regressive generation
        
    def generate_text(self, prompt, max_new_tokens=1024, temperature=0.7, top_k=50, top_p=0.9):
        self.eval()
        device = next(self.parameters()).device

        prompt_str = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        
        input_len = list(input_ids.shape)
        
        compressed_kvs = None

        if input_len[1] >= 1024:
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                x_factor = 16
                # ---------------------------------------------------------
                # STEP 1: Extract Raw KV Cache of the past
                # ---------------------------------------------------------
                print("\n" + "="*50)
                print(" STEP 1: EXTRACTING & COMPRESSING KV CACHES")
                print("="*50)
                K_latent_len = (input_len[1] + x_factor -1) // x_factor
                
                print("Passing long context through AR model to extract raw KVs...")
                past_outputs = self.ar_model(input_ids)
                uncompressed_kvs = past_outputs["active_kv_caches"]
                
                print(f"🟢 Original KV Cache Shape: {uncompressed_kvs.shape}")

                # ---------------------------------------------------------
                # STEP 2: Compress the KV Caches via DC
                # ---------------------------------------------------------
                print(f"\nPassing raw KVs through DC Model for 16x compression...")
                # if No latent_states, we assume, it is the first step and so we just put a noisy latent
                # and also make start_step = 0
                start_step = 0
                latent_states = torch.randn(
                    1, 
                    K_latent_len, 
                    1152, 
                    device=device, 
                    dtype=torch.bfloat16
                )

                # Compress the extracted KVs using the DM tower 
                T_latent_states, T_compressed_kvs = self.kv_compress(latent_states, uncompressed_kvs, start_step, 2, 0)
            
                compressed_kvs = T_compressed_kvs[:, -1, ...]
                ck_shape = compressed_kvs.shape
                print(f"🔵 Compressed KV Cache Shape : {compressed_kvs.shape}")
                print(f"   -> Successfully compressed to just {ck_shape[2]} tokens of memory space!")


                # refreshes input_ids to just a single bos:
                input_ids = torch.tensor([[2]]).to(device)


        
        # Auto Regressive generations:

        # EOS and Turn-end tokens for Gemma
        eos_token_ids = [self.tokenizer.eos_token_id, 106] 

        print(f"\nUser: {prompt}\nModel: ", end="", flush=True)

        active_kv_caches = None

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.ar_model(
                    input_ids, 
                    active_kv_caches=active_kv_caches, 
                    compressed_kv_caches=compressed_kvs
                )
                
            logits = outputs["logits"]
            active_kv_caches = outputs["active_kv_caches"]
            
            # Get the logits for the last token in the sequence
            next_token_logits = logits[:, -1, :]
            
            # --- MULTINOMIAL SAMPLING LOGIC ---
            
            # 1. Apply Temperature
            if temperature != 1.0 and temperature > 0.0:
                next_token_logits = next_token_logits / temperature
                
            # 2. Apply Top-K filtering
            if top_k > 0:
                # Find the top k values and their indices
                top_k_values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                # Mask out everything below the k-th highest value
                indices_to_remove = next_token_logits < top_k_values[:, -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # 3. Apply Top-P (Nucleus) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1, dtype=torch.float32), dim=-1)

                # Remove tokens with cumulative probability above the threshold (top_p)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter the mask back to the original indices
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')

            # 4. Convert to probabilities (use float32 for numerical stability)
            probs = F.softmax(next_token_logits, dim=-1, dtype=torch.float32)

            # 5. Sample the next token
            if temperature == 0.0:
                # Fallback to greedy if temperature is strictly 0
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Multinomial sample
                next_token = torch.multinomial(probs, num_samples=1)
                
            # ----------------------------------

            # Only pass the new token for the next iteration
            input_ids = next_token  
            
            # Decode and print dynamically
            print(self.tokenizer.decode([next_token.item()]), end="", flush=True)

            # Check for End of Sequence
            if next_token.item() in eos_token_ids:
                break
                
        print("\n")






"""
We are asking CompDiTModel to produce best latent vector which will act as query 
which then going to look at the LLM KVs and produce best latent KVS as compressed KVS to be used by LLM for further autoregression;

I doubt it!

Instead; we should keep the prev. step latent KVs and tell the diffusion model to reason onto itself what he know and what he need to know more
which will produce best KVs to act as compressed KVs...

Just Implemented...
"""



# device = 'cpu'


# ILDC_model = ILDC(device = device)
# ILDC_model.to(torch.bfloat16)

# AR_model = ILDC_model.ar_model

# with torch.no_grad(): 
#     # [B, Seq_len]
#     # input_ids = None
#     input_ids = torch.randint(1, 26400, (1, 16))
    
#     # [B, Layers, Seq_len, KV, heads, Head_dim]
#     active_kv_caches = None
#     # active_kv_caches = torch.randn(1, 26, 1024, 2, 1, 256).to(torch.bfloat16)

#     # compressed_kv_caches = None
#     compressed_kv_caches = torch.randn(1, 26, 15, 2, 1, 256).to(torch.bfloat16)
#     outputs = AR_model(input_ids, active_kv_caches, compressed_kv_caches)
#     print(outputs["hidden_states"].shape)
#     print(outputs["active_kv_caches"].shape)



# DC_model = CompDiTModel().to(torch.bfloat16)

# with torch.no_grad(): 
#     # [B, Layers, Seq_len, KV, heads, Head_dim]
#     kv_caches = torch.randn(3, 26, 1024, 2, 1, 256).to(torch.bfloat16)
#     latent_states = torch.randn(3, 64, 1152).to(torch.bfloat16)
#     A, B = DC_model(latent_states, kv_caches, 1)
#     print(A, B)


# device = 'cuda'

# ILDC_model = ILDC(device = device)
# ILDC_model.to(torch.bfloat16)
# with torch.no_grad():
#     start_step = 0
#     diff_steps = 2
#     total_diff_steps = 8
    
#     for _ in range((total_diff_steps//diff_steps -1)): 

#         train_output = ILDC_model(
#             input_ids = torch.randint(1, 26400, (3, 512)).to(device), 
#             context_ids=torch.randint(1, 26400, (3, 2048)).to(device), 
#             # active_kv_caches= torch.randn(3, 26, 1024, 2, 1, 256).to(torch.bfloat16).to(device), 
#             # active_compressed_kv= torch.randn(3, 26, 768, 2, 1, 256).to(torch.bfloat16).to(device), 
#             # start_pos=768,  # as we already have 768 compressed kvs

#             start_step = start_step, 
#             diff_steps= diff_steps,
#         )
#         start_step += diff_steps

#         print(train_output)




# hidden_fact = "The secret code to bypass the mainframe is 'OMEGA-77'."
# filler_text = "The system logs show normal operational status with minor fluctuations in the thermal array. " * 100
# question = "What's the secret code of the mainframe? Hello"
# prompt = filler_text + hidden_fact + filler_text 

# ILDC_model.generate_text(prompt)