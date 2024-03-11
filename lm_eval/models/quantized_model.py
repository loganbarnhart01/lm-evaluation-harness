from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.models.hugginface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype

import torch
import accelerate
import bitsandbytes

from peft import (
  LoraConfig,
  get_peft_model,
)

from transformers import (
  AutoModelForCausalLM,
  BitsAndBytesConfig,
)

eval_logger = utils.eval_logger

@register_model('qlora')
class QuantizedHFLM(HFLM):
    def __init__(self, *args, quantized=False, **kwargs):
        self.quantized = quantized  # Save the new argument as an instance variable
        super().__init__(*args, **kwargs)  # Correctly call the superclass's __init__

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: Optional[bool] = False,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        # PEFT and quantization options
        peft: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        **kwargs,):
        if self.quantized:
            if self.AUTO_MODEL_CLASS == AutoModelForCausalLM:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias='none',
                    task_type='causal_LM'
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    pretrained,
                    torch_dtype=get_dtype(dtype),
                    trust_remote_code=trust_remote_code,
                    quantization_config=bnb_config,
                )
                self._model = get_peft_model(self._model, lora_config)
            else:
                super()._create_model(pretrained, revision, dtype, trust_remote_code, parallelize, device_map_option, max_memory_per_gpu, max_cpu_memory, offload_folder, peft, autogptq, **kwargs,)
        else:
            # Call the superclass method if you want to reuse its logic unchanged
            # when 'quantized' is None
            super()._create_model(pretrained, revision, dtype, trust_remote_code, parallelize, device_map_option, max_memory_per_gpu, max_cpu_memory, offload_folder, peft, autogptq, **kwargs,)