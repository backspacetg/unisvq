from transformers.models.qwen3 import Qwen3ForCausalLM, Qwen3Config
from .tvq import TVQHadLinear

class Qwen3TVQConfig(Qwen3Config):
    model_type = "tvqqwen3"
    def __init__(self, *args, vec_size=4, vec_bit=2, group_size=128, proj_type="bypass", use_hadamard=False, codebook_trainable=False, **kwargs):
        self.vec_size = vec_size
        self.vec_bit = vec_bit
        self.group_size = group_size
        self.proj_type = proj_type
        self.use_hadamard = use_hadamard
        self.codebook_trainable = codebook_trainable
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
        group_size = config.group_size
        vec_size = config.vec_size
        vec_bit = config.vec_bit
        proj_type = config.proj_type
        code_book_args = {"group_size": group_size, "vec_size": vec_size, "vec_bit": vec_bit, "proj_type": proj_type, 'codebook_trainable': config.codebook_trainable}
        attention_bias = getattr(config, "attention_bias", True)
        for layer in self.model.layers:
            layer.self_attn.q_proj = TVQHadLinear(hidden_size, num_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.k_proj = TVQHadLinear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.v_proj = TVQHadLinear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias, **code_book_args)
            layer.self_attn.o_proj = TVQHadLinear(num_heads * head_dim, hidden_size, bias=False, **code_book_args)
            layer.mlp.gate_proj = TVQHadLinear(hidden_size, intermediate_size, bias=False, **code_book_args)
            layer.mlp.up_proj = TVQHadLinear(hidden_size, intermediate_size, bias=False, **code_book_args)
            layer.mlp.down_proj = TVQHadLinear(intermediate_size, hidden_size, bias=False, **code_book_args)                
        self.config = config
    
    def freeze_model(self):
        self.model.norm.weight.requires_grad = False
        for layer in self.model.layers:
            layer.input_layernorm.weight.requires_grad = False
            layer.post_attention_layernorm.weight.requires_grad = False
            layer.self_attn.k_norm.weight.requires_grad = False
            layer.self_attn.q_norm.weight.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False
        self.model.embed_tokens.weight.requires_grad = False