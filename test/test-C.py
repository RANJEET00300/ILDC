import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


from model.helper_fn import RMSNorm, apply_rotary_pos_emb, get_timestep_embedding
from model.config import ModelConfig, TrainingConfig

from model.AR_Model import CausalLM
from model.DC_Model import CompDiTModel
from model.ILDC_Model import ILDC



def test():
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ILDC_model = ILDC(device = device)
    ILDC_model.to(torch.bfloat16)

    AR_model = ILDC_model.ar_model

    with torch.no_grad(): 
        # [B, Seq_len]
        # input_ids = None
        input_ids = torch.randint(1, 26400, (1, 16))
        
        # [B, Layers, Seq_len, KV, heads, Head_dim]
        active_kv_caches = None
        # active_kv_caches = torch.randn(1, 26, 1024, 2, 1, 256).to(torch.bfloat16)

        # compressed_kv_caches = None
        compressed_kv_caches = torch.randn(1, 26, 15, 2, 1, 256).to(torch.bfloat16)
        outputs = AR_model(input_ids, active_kv_caches, compressed_kv_caches)
        print(outputs["hidden_states"].shape)
        print(outputs["active_kv_caches"].shape)



    DC_model = ILDC_model.dc_model

    with torch.no_grad(): 
        # [B, Layers, Seq_len, KV, heads, Head_dim]
        kv_caches = torch.randn(3, 26, 1024, 2, 1, 256).to(torch.bfloat16)
        latent_states = torch.randn(3, 64, 1152).to(torch.bfloat16)
        A, B = DC_model(latent_states, kv_caches, 1)
        print(A, B)


    # ILDC Model
    with torch.no_grad():
        start_step = 0
        diff_steps = 2
        total_diff_steps = 8
        
        for _ in range((total_diff_steps//diff_steps -1)): 

            train_output = ILDC_model(
                input_ids = torch.randint(1, 26400, (3, 512)).to(device), 
                context_ids=torch.randint(1, 26400, (3, 2048)).to(device), 
                # active_kv_caches= torch.randn(3, 26, 1024, 2, 1, 256).to(torch.bfloat16).to(device), 
                # active_compressed_kv= torch.randn(3, 26, 768, 2, 1, 256).to(torch.bfloat16).to(device), 
                # start_pos=768,  # as we already have 768 compressed kvs

                start_step = start_step, 
                diff_steps= diff_steps,
            )
            start_step += diff_steps

            print(train_output)




    # Autoregressive Generation with Compressed KVs

    hidden_fact = "The secret code to bypass the mainframe is 'OMEGA-77'."
    filler_text = "The system logs show normal operational status with minor fluctuations in the thermal array. " * 100
    question = "What's the secret code of the mainframe? Hello"
    prompt = filler_text + hidden_fact + filler_text 

    ILDC_model.generate_text(prompt)



def main():
    print("="*30)
    print("Starting the Test")
    
    test()
    
    print("="*30)
    print("Test Completed")

if __name__ == "__main__":
    main()
