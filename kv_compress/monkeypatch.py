from importlib.metadata import version
import warnings
from transformers.models import llama, mistral

from kv_compress.segmentkv.llama_hijack import llama_flash_attn2_forward as segment_llama_forward
from .segmentkv.mistral_hijack import mistral_flash_attn2_forward as segment_mistral_forward

from .snapkv.llama_hijack import llama_flash_attn2_forward as snap_llama_forward
from .snapkv.mistral_hijack import mistral_flash_attn2_forward as snap_mistral_forward

from .chunkkv.llama_hijack import llama_flash_attn2_forward as chunk_llama_forward
from .chunkkv.mistral_hijack import mistral_flash_attn2_forward as chunk_mistral_forward


def check_version():
    transformers_version = None
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    version_list = ['4.56.2']  ### 4.56.2 now
    warning_flag = True
    for ver in version_list:
        if ver in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(
            f"Transformers version {transformers_version} might not be compatible with kv_comp. kv_comp is tested with Transformers version {version_list}.")


def replace_llama_by_segmentkv():
    check_version()
    print('replace_llama_by_segmentkv')
    llama.modeling_llama.LlamaAttention.forward = segment_llama_forward


def replace_llama_by_segmentkv_two_stage():
    check_version()
    print('replace_llama_by_segmentkv_two_stage')

    from kv_compress.segmentkv.llama_hijack_two_stages import llama_flash_attn2_forward as two_stage_segment_llama_forward
    llama.modeling_llama.LlamaAttention.forward = two_stage_segment_llama_forward


def replace_mistral_by_segmentkv():
    check_version()
    print('replace_mistral_by_segmentkv')
    mistral.modeling_mistral.MistralAttention.forward = segment_mistral_forward

def replace_llama_by_snapkv():
    check_version()
    print('replace_llama_by_snapkv')
    llama.modeling_llama.LlamaAttention.forward = snap_llama_forward


def replace_mistral_by_snapkv():
    check_version()
    print('replace_mistral_by_snapkv')
    mistral.modeling_mistral.MistralAttention.forward = snap_mistral_forward


def replace_llama_by_chunkkv():
    check_version()
    print('replace_llama_by_chunkkv')
    llama.modeling_llama.LlamaAttention.forward = chunk_llama_forward


def replace_mistral_by_chunkkv():
    check_version()
    print('replace_mistral_by_chunkkv')
    mistral.modeling_mistral.MistralAttention.forward = chunk_mistral_forward


def replace_llama_by_laqkv():
    check_version()
    print('replace_llama_by_laqkv')
    from kv_compress.LAQ.llama_model import llama_flash_attn2_forward_LAQ, LlamaForCausalLM_forward_LAQ
    llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_LAQ
    llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_forward_LAQ

