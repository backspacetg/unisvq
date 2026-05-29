from transformers.models.qwen3 import Qwen3ForCausalLM, Qwen3Config
from .tvq import TVQHadLinear

class Qwen3TVQConfig(Qwen3Config):
    model_type = "qwen3_tvq"
    def __init__(self, *args, vec_size=4, vec_bit=2, proj_type="bypass",  mixed_percision=False, **kwargs):
        self.vec_size = vec_size
        self.vec_bit = vec_bit
        self.proj_type = proj_type
        self.mixed_percision = mixed_percision
        super().__init__(*args, **kwargs)
    

class Qwen3TVQForCausalLM(Qwen3ForCausalLM):
    config_class = Qwen3TVQConfig
    def __init__(self, config: Qwen3TVQConfig):
        super().__init__(config)
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", hidden_size//num_heads)
        num_key_value_heads = config.num_key_value_heads
        intermediate_size = config.intermediate_size
        vec_size = config.vec_size
        vec_bit = config.vec_bit
        proj_type = config.proj_type
        code_book_args = {
            "vec_size": vec_size, 
            "vec_bit": vec_bit, 
            "proj_type": proj_type, 
            "mixed_percision": config.mixed_percision
        }
        attention_bias = getattr(config, "attention_bias", False)
        for layer in self.model.layers:
            layer.self_attn.q_proj = TVQHadLinear(hidden_size, num_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.k_proj = TVQHadLinear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.v_proj = TVQHadLinear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.o_proj = TVQHadLinear(num_heads * head_dim, hidden_size, bias=False, **code_book_args)
            layer.mlp.gate_proj = TVQHadLinear(hidden_size, intermediate_size, bias=False, **code_book_args)
            layer.mlp.up_proj = TVQHadLinear(hidden_size, intermediate_size, bias=False, **code_book_args)
            layer.mlp.down_proj = TVQHadLinear(intermediate_size, hidden_size, bias=False, **code_book_args)                
        self.config = config
        self.mask_rate = 0.0
