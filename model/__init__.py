from transformers.models.auto import AutoModel, AutoConfig, AutoModelForCausalLM
from .modeling_megatron_llama import MegaTVQConfig, MegaTVQForCausalLM, MegaTVQInferForCausalLM, MegaTVQInferConfig
from .modeling_qwen3_vq import Qwen3TVQConfig, Qwen3TVQForCausalLM
from .modeling_9g import FM9G7BTVQInferConfig, FM9G7BTVQInferForCausalLM

AutoConfig.register('tvq_fm9g7b_infer', FM9G7BTVQInferConfig)
AutoModel.register(FM9G7BTVQInferConfig, FM9G7BTVQInferForCausalLM)
AutoModelForCausalLM.register(FM9G7BTVQInferConfig, FM9G7BTVQInferForCausalLM)

AutoConfig.register('mega_tvq_infer', MegaTVQInferConfig)
AutoModel.register(MegaTVQInferConfig, MegaTVQInferForCausalLM)
AutoModelForCausalLM.register(MegaTVQInferConfig, MegaTVQInferForCausalLM)

AutoConfig.register('mega_tvq', MegaTVQConfig)
AutoModel.register(MegaTVQConfig, MegaTVQForCausalLM)
AutoModelForCausalLM.register(MegaTVQConfig, MegaTVQForCausalLM)

AutoConfig.register('qwen3_tvq', Qwen3TVQConfig)
AutoModel.register(Qwen3TVQConfig, Qwen3TVQForCausalLM)
AutoModelForCausalLM.register(Qwen3TVQConfig, Qwen3TVQForCausalLM)
