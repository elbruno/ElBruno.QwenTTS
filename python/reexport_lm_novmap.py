"""Re-export all LM ONNX models with vmap-free masking for C# ORT compatibility."""

import torch
import torch.nn as nn
import os
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.executorch import sdpa_mask_without_vmap
from transformers.cache_utils import DynamicCache

# Register vmap-free functions (ExecuTorch approach)
ALL_MASK_ATTENTION_FUNCTIONS.register('sdpa_without_vmap', sdpa_mask_without_vmap)
ALL_ATTENTION_FUNCTIONS.register('sdpa_without_vmap', ALL_ATTENTION_FUNCTIONS['sdpa'])

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

TALKER_NUM_LAYERS = 28
TALKER_NUM_KV_HEADS = 8
TALKER_HEAD_DIM = 128
TALKER_HIDDEN = 1024

CP_NUM_LAYERS = 5
CP_NUM_KV_HEADS = 8
CP_HEAD_DIM = 128
CP_HIDDEN = 1024
CP_NUM_GROUPS = 15

OPSET_VERSION = 17


class TalkerPrefillWrapper(nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head

    def forward(self, inputs_embeds, attention_mask, position_ids):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states[:, -1:, :])
        cache = outputs.past_key_values
        flat_kv = []
        for i in range(TALKER_NUM_LAYERS):
            flat_kv.append(cache.layers[i].keys)
            flat_kv.append(cache.layers[i].values)
        return (logits, hidden_states, *flat_kv)


class TalkerDecodeWrapper(nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head

    def forward(self, inputs_embeds, attention_mask, position_ids, past_keys, past_values):
        cache = DynamicCache()
        for i in range(TALKER_NUM_LAYERS):
            cache.update(past_keys[i], past_values[i], i)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=cache,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)
        new_cache = outputs.past_key_values
        present_keys = torch.stack([new_cache.layers[i].keys for i in range(TALKER_NUM_LAYERS)])
        present_values = torch.stack([new_cache.layers[i].values for i in range(TALKER_NUM_LAYERS)])
        return (logits, hidden_states, present_keys, present_values)


class CodePredictorWrapper(nn.Module):
    def __init__(self, code_predictor):
        super().__init__()
        self.model = code_predictor.model
        self.projection = code_predictor.small_to_mtp_projection
        all_weights = torch.stack([head.weight for head in code_predictor.lm_head])
        self.register_buffer("lm_head_weights", all_weights)

    def forward(self, inputs_embeds, generation_steps, past_keys, past_values):
        inputs_embeds = self.projection(inputs_embeds)
        cache = DynamicCache()
        for i in range(CP_NUM_LAYERS):
            cache.update(past_keys[i], past_values[i], i)
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=cache,
        )
        hidden_states = outputs.last_hidden_state
        weight = torch.index_select(self.lm_head_weights, 0, generation_steps.view(-1))
        weight = weight.squeeze(0)
        logits = torch.matmul(hidden_states, weight.t())
        new_cache = outputs.past_key_values
        present_keys = torch.stack([new_cache.layers[i].keys for i in range(CP_NUM_LAYERS)])
        present_values = torch.stack([new_cache.layers[i].values for i in range(CP_NUM_LAYERS)])
        return (logits, present_keys, present_values)


def patch_model_for_vmap_free(model):
    """Patch all sub-models to use vmap-free masking."""
    # Talker
    model.talker.model.config._attn_implementation = 'sdpa_without_vmap'
    for layer in model.talker.model.layers:
        layer.self_attn.config._attn_implementation = 'sdpa_without_vmap'
    # Code Predictor
    model.talker.code_predictor.model.config._attn_implementation = 'sdpa_without_vmap'
    for layer in model.talker.code_predictor.model.layers:
        layer.self_attn.config._attn_implementation = 'sdpa_without_vmap'


def fix_external_data_ref(onnx_path):
    """Ensure external data references use the correct filename."""
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    expected_data = os.path.basename(onnx_path) + '.data'
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == 'location':
                entry.value = expected_data
    onnx.save(model, onnx_path)


def export_prefill(talker, output_dir):
    print("Exporting talker_prefill.onnx ...")
    wrapper = TalkerPrefillWrapper(talker).eval()
    B, T = 1, 8
    dummy = (
        torch.randn(B, T, TALKER_HIDDEN),
        torch.ones(B, T, dtype=torch.int64),
        torch.zeros(3, B, T, dtype=torch.int64),
    )
    input_names = ['inputs_embeds', 'attention_mask', 'position_ids']
    output_names = ['logits', 'hidden_states']
    for i in range(TALKER_NUM_LAYERS):
        output_names.extend([f'present_key_{i}', f'present_value_{i}'])
    dynamic_axes = {
        'inputs_embeds': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'position_ids': {1: 'batch_size', 2: 'sequence_length'},
        'logits': {0: 'batch_size'},
        'hidden_states': {0: 'batch_size', 1: 'sequence_length'},
    }
    for i in range(TALKER_NUM_LAYERS):
        dynamic_axes[f'present_key_{i}'] = {0: 'batch_size', 2: 'sequence_length'}
        dynamic_axes[f'present_value_{i}'] = {0: 'batch_size', 2: 'sequence_length'}

    path = os.path.join(output_dir, 'talker_prefill.onnx')
    torch.onnx.export(wrapper, dummy, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=OPSET_VERSION,
        do_constant_folding=True)
    fix_external_data_ref(path)
    print("  Done")


def export_decode(talker, output_dir):
    print("Exporting talker_decode.onnx ...")
    wrapper = TalkerDecodeWrapper(talker).eval()
    B, T_past = 1, 8
    dummy = (
        torch.randn(B, 1, TALKER_HIDDEN),
        torch.ones(B, T_past + 1, dtype=torch.int64),
        torch.zeros(3, B, 1, dtype=torch.int64),
        torch.randn(TALKER_NUM_LAYERS, B, TALKER_NUM_KV_HEADS, T_past, TALKER_HEAD_DIM),
        torch.randn(TALKER_NUM_LAYERS, B, TALKER_NUM_KV_HEADS, T_past, TALKER_HEAD_DIM),
    )
    input_names = ['inputs_embeds', 'attention_mask', 'position_ids', 'past_keys', 'past_values']
    output_names = ['logits', 'hidden_states', 'present_keys', 'present_values']
    dynamic_axes = {
        'inputs_embeds': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size', 1: 'total_sequence_length'},
        'position_ids': {1: 'batch_size'},
        'logits': {0: 'batch_size'},
        'hidden_states': {0: 'batch_size'},
        'past_keys': {1: 'batch_size', 3: 'past_sequence_length'},
        'past_values': {1: 'batch_size', 3: 'past_sequence_length'},
        'present_keys': {1: 'batch_size', 3: 'total_sequence_length'},
        'present_values': {1: 'batch_size', 3: 'total_sequence_length'},
    }
    path = os.path.join(output_dir, 'talker_decode.onnx')
    torch.onnx.export(wrapper, dummy, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=OPSET_VERSION,
        do_constant_folding=True)
    fix_external_data_ref(path)
    print("  Done")


def export_code_predictor(talker, output_dir):
    print("Exporting code_predictor.onnx ...")
    wrapper = CodePredictorWrapper(talker.code_predictor).eval()
    B, S, T_past = 1, 2, 2
    dummy = (
        torch.randn(B, S, CP_HIDDEN),
        torch.tensor([0], dtype=torch.int64),
        torch.randn(CP_NUM_LAYERS, B, CP_NUM_KV_HEADS, T_past, CP_HEAD_DIM),
        torch.randn(CP_NUM_LAYERS, B, CP_NUM_KV_HEADS, T_past, CP_HEAD_DIM),
    )
    input_names = ['inputs_embeds', 'generation_steps', 'past_keys', 'past_values']
    output_names = ['logits', 'present_keys', 'present_values']
    dynamic_axes = {
        'inputs_embeds': {0: 'batch_size', 1: 'sequence_length'},
        'generation_steps': {0: 'num_steps'},
        'past_keys': {1: 'batch_size', 3: 'past_sequence_length'},
        'past_values': {1: 'batch_size', 3: 'past_sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'},
        'present_keys': {1: 'batch_size', 3: 'total_sequence_length'},
        'present_values': {1: 'batch_size', 3: 'total_sequence_length'},
    }
    path = os.path.join(output_dir, 'code_predictor.onnx')
    # Code predictor needs dynamo=False due to index_select + t() pattern
    torch.onnx.export(wrapper, dummy, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=OPSET_VERSION,
        do_constant_folding=True, dynamo=False)
    fix_external_data_ref(path)
    print("  Done")


if __name__ == '__main__':
    print("Loading model...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        'python/models/Qwen3-TTS-0.6B-CustomVoice',
        dtype=torch.float32,
        attn_implementation='sdpa',
    )
    model.eval()
    patch_model_for_vmap_free(model)
    
    output_dir = 'python/onnx_runtime'
    export_prefill(model.talker, output_dir)
    export_decode(model.talker, output_dir)
    export_code_predictor(model.talker, output_dir)
    print("\nAll models exported successfully!")

