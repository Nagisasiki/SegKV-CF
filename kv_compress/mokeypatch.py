from transformers.models import llama, mistral

from kv_compress.snapkv.llama_hijack import llama_flash_attn2_forward_snap
from kv_compress.chunkkv.llama_hijack import llama_flash_attn2_forward_chunk
from kv_compress.laq.llama_model import llama_flash_attn2_forward_LAQ, LlamaForCausalLM_forward_LAQ
from kv_compress.segkv.llama_hijack import llama_flash_attn2_forward_segkv_cf, llama_flash_attn2_forward_seg

from kv_compress.snapkv.mistral_hijack import mistral_flash_attn2_forward_snap
from kv_compress.chunkkv.mistral_hijack import mistral_flash_attn2_forward_chunk
from kv_compress.laq.mistral_model import mistral_flash_attn2_forward_LAQ, MistralForCausalLM_forward_LAQ
from kv_compress.segkv.mistral_hijack import mistral_flash_attn2_forward_segkv_cf, mistral_flash_attn2_forward_seg



def replace_llama(method):

    if method == "SnapKV":
        print('replace_llama_by_SnapKV')
        llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_snap

    if method == "ChunkKV":
        print('replace_llama_by_ChunkKV')
        llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_chunk

    if method == "LAQ":
        print('replace_llama_by_LAQ')
        llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_LAQ
        llama.modeling_llama.LlamaForCausalLM.forward = LlamaForCausalLM_forward_LAQ

    if method == "SegKV":
        print('replace_llama_by_SegKV')
        llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_seg

    if method == "SegKV_CF":
        print('replace_llama_by_SegKV_CF')
        llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward_segkv_cf


def replace_mistral(method: str):
    if method == "SnapKV":
        print("replace_mistral_by_SnapKV")
        mistral.modeling_mistral.MistralAttention.forward = mistral_flash_attn2_forward_snap

    if method == "ChunkKV":
        print("replace_mistral_by_ChunkKV")
        mistral.modeling_mistral.MistralAttention.forward = mistral_flash_attn2_forward_chunk

    if method == "LAQ":
        print("replace_mistral_by_LAQ")
        mistral.modeling_mistral.MistralAttention.forward = mistral_flash_attn2_forward_LAQ
        mistral.modeling_mistral.MistralForCausalLM.forward = MistralForCausalLM_forward_LAQ

    if method == "SegKV":
        print("replace_mistral_by_SegKV")
        mistral.modeling_mistral.MistralAttention.forward = mistral_flash_attn2_forward_seg

    if method == "SegKV_CF":
        print("replace_mistral_by_SegKV_CF")
        mistral.modeling_mistral.MistralAttention.forward = mistral_flash_attn2_forward_segkv_cf


