"""
Patch ONNX models for DirectML compatibility.

DirectML's Reshape operator doesn't support -1 (inferred dimension) when combined
with dynamic dimensions from Shape ops. This script replaces -1 in each Reshape
node's shape with the actual value computed from the producing MatMul's weight dims.

When multiple Reshape nodes share the same shape tensor but need different -1 values,
the script creates per-node shape tensors with the correct resolved values.

Model architecture (from config.json):
  - Talker: 28 layers, 16 attn heads, 8 KV heads, head_dim=128, hidden=1024
  - Code Predictor: 5 layers, 16 attn heads, 8 KV heads, head_dim=128

Usage:
    python patch_models_for_dml.py <model_dir>
    python patch_models_for_dml.py python/onnx_runtime
"""

import argparse
import json
import os
import numpy as np
import onnx
from onnx import TensorProto, helper


def get_initializer_map(model):
    """Build map of initializer name -> numpy array (int64 only)."""
    result = {}
    for init in model.graph.initializer:
        if init.data_type == TensorProto.INT64:
            result[init.name] = np.frombuffer(init.raw_data, dtype=np.int64).copy()
    return result


def get_node_producing(model, output_name):
    """Find the node that produces the given output tensor."""
    for node in model.graph.node:
        if output_name in node.output:
            return node
    return None


def trace_to_matmul_weight_dim(model, tensor_name, depth=0):
    """Trace backwards from a tensor to find a MatMul weight's output dimension."""
    if depth > 15:
        return None

    node = get_node_producing(model, tensor_name)
    if node is None:
        return None

    if node.op_type == "MatMul":
        weight_name = node.input[1]
        for init in model.graph.initializer:
            if init.name == weight_name:
                return list(init.dims)[-1]
        return None

    # Follow through common ops that preserve the last dimension
    passthrough_ops = {"Add", "Mul", "Sub", "Div", "Relu", "Gelu", "Tanh",
                       "Sigmoid", "Cast", "Identity", "Reciprocal",
                       "Sqrt", "Pow"}
    if node.op_type in passthrough_ops:
        for inp in node.input:
            result = trace_to_matmul_weight_dim(model, inp, depth + 1)
            if result is not None:
                return result

    return None


def analyze_reshape_shape(model, reshape_node, init_map):
    """Analyze a Reshape node's shape input. Returns (has_neg1, known_dims_product)
    where known_dims_product is the product of all non-negative, non-dynamic dimensions."""
    shape_name = reshape_node.input[1]

    # Direct initializer
    if shape_name in init_map:
        arr = init_map[shape_name]
        if -1 not in arr:
            return False, None, None
        known = [v for v in arr if v > 0]
        pattern = arr.tolist()
        return True, int(np.prod(known)) if known else 1, pattern

    # Via Concat
    concat_node = get_node_producing(model, shape_name)
    if concat_node is None or concat_node.op_type != "Concat":
        return False, None, None

    pattern = []
    known_product = 1
    has_neg1 = False
    for inp in concat_node.input:
        if inp in init_map:
            for v in init_map[inp]:
                pattern.append(int(v))
                if v == -1:
                    has_neg1 = True
                elif v > 0:
                    known_product *= v
        else:
            pattern.append("?")

    return has_neg1, int(known_product), pattern


def load_model_config(model_dir):
    """Load architecture config for fallback -1 resolution."""
    config_path = os.path.join(model_dir, "embeddings", "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return None


def get_merge_dim_from_config(config):
    """Get the attention output merge dimension (num_q_heads * head_dim) from config."""
    if config is None:
        return None
    talker = config.get("talker", {})
    num_heads = talker.get("num_attention_heads", 16)
    head_dim = talker.get("head_dim", 128)
    return num_heads * head_dim


def patch_model(model_path, output_path=None, config=None):
    """Patch a single ONNX model to replace -1 in Reshape shapes."""
    if output_path is None:
        output_path = model_path

    # Only use config-based fallback for talker models
    fname = os.path.basename(model_path).lower()
    is_talker = "talker" in fname
    effective_config = config if is_talker else None

    print(f"\nPatching: {model_path}")
    model = onnx.load(model_path, load_external_data=False)
    init_map = get_initializer_map(model)

    patched_count = 0
    skipped_count = 0
    new_init_counter = [0]

    def make_shape_init(shape_values, prefix="dml_shape"):
        """Create a new initializer with the given shape values."""
        name = f"{prefix}_{new_init_counter[0]}"
        new_init_counter[0] += 1
        arr = np.array(shape_values, dtype=np.int64)
        tensor = helper.make_tensor(name, TensorProto.INT64, [len(arr)], arr)
        model.graph.initializer.append(tensor)
        return name

    # Pass 1: Replace Reshape(x, [-1, 1]) with Unsqueeze(x, axis=1)
    # This handles causal mask construction where arange is reshaped for broadcasting
    nodes_to_remove = []
    nodes_to_add = []
    for i, node in enumerate(model.graph.node):
        if node.op_type != "Reshape":
            continue
        shape_name = node.input[1]
        if shape_name in init_map:
            arr = init_map[shape_name]
            if list(arr) == [-1, 1]:
                unsqueeze_axes = make_shape_init([1], f"dml_unsq_axes_{node.name}")
                unsqueeze = helper.make_node(
                    "Unsqueeze",
                    inputs=[node.input[0], unsqueeze_axes],
                    outputs=list(node.output),
                    name=f"dml_unsqueeze_{node.name}"
                )
                nodes_to_remove.append(node)
                nodes_to_add.append((i, unsqueeze))
                patched_count += 1
                print(f"  Patched {node.name}: Reshape([-1,1]) -> Unsqueeze(axis=1)")

    for node in nodes_to_remove:
        model.graph.node.remove(node)
    for idx, new_node in sorted(nodes_to_add, key=lambda x: x[0], reverse=True):
        model.graph.node.insert(idx, new_node)

    # Pass 1b: Convert 1D ConvTranspose to 2D for DML compatibility.
    # DML often fails on 1D (3D tensor) ConvTranspose. The workaround is:
    # Unsqueeze(input, axis=2) → Reshape weight to 4D → 2D ConvTranspose → Squeeze(output, axis=2)
    ct_nodes_to_remove = []
    ct_nodes_to_add = []
    for i, node in enumerate(model.graph.node):
        if node.op_type != "ConvTranspose":
            continue
        weight_name = node.input[1]
        weight_init = None
        for init in model.graph.initializer:
            if init.name == weight_name:
                weight_init = init
                break
        if weight_init is None:
            continue
        weight_dims = list(weight_init.dims)
        if len(weight_dims) != 3:
            continue  # only patch 1D ConvTranspose (3D weight: in_ch, out_ch, kernel)

        # Reshape weight from [in_ch, out_ch, K] to [in_ch, out_ch, 1, K] using Reshape node
        new_weight_shape_name = make_shape_init(
            [weight_dims[0], weight_dims[1], 1, weight_dims[2]],
            f"dml_ct_wshape_{node.name}"
        )
        reshaped_weight_name = f"dml_ct_w4d_{node.name}"
        weight_reshape_node = helper.make_node(
            "Reshape", inputs=[weight_name, new_weight_shape_name],
            outputs=[reshaped_weight_name], name=f"dml_reshape_ct_w_{node.name}"
        )

        # Build 2D ConvTranspose attributes from 1D attrs
        new_attrs = []
        for attr in node.attribute:
            if attr.name in ("kernel_shape", "strides", "dilations"):
                new_attrs.append(helper.make_attribute(attr.name, [1] + list(attr.ints)))
            elif attr.name == "output_padding":
                new_attrs.append(helper.make_attribute(attr.name, [0] + list(attr.ints)))
            elif attr.name == "pads":
                p = list(attr.ints)
                if len(p) == 2:
                    new_attrs.append(helper.make_attribute("pads", [0, p[0], 0, p[1]]))
                else:
                    new_attrs.append(attr)
            else:
                new_attrs.append(attr)
        if not any(a.name == "kernel_shape" for a in new_attrs):
            new_attrs.append(helper.make_attribute("kernel_shape", [1, weight_dims[2]]))

        # Unsqueeze input: (N, C, L) → (N, C, 1, L)
        unsq_axes = make_shape_init([2], f"dml_ct_unsq_axes_{node.name}")
        unsq_out = f"dml_ct_unsq_{node.name}"
        unsqueeze_node = helper.make_node(
            "Unsqueeze", inputs=[node.input[0], unsq_axes],
            outputs=[unsq_out], name=f"dml_unsqueeze_ct_{node.name}"
        )

        # 2D ConvTranspose
        ct2d_out = f"dml_ct2d_out_{node.name}"
        ct2d_inputs = [unsq_out, reshaped_weight_name] + list(node.input[2:])
        ct2d_node = helper.make_node(
            "ConvTranspose", inputs=ct2d_inputs, outputs=[ct2d_out],
            name=f"dml_ct2d_{node.name}"
        )
        ct2d_node.attribute.clear()
        for a in new_attrs:
            ct2d_node.attribute.append(a)

        # Squeeze output: (N, C, 1, L') → (N, C, L')
        sq_axes = make_shape_init([2], f"dml_ct_sq_axes_{node.name}")
        squeeze_node = helper.make_node(
            "Squeeze", inputs=[ct2d_out, sq_axes],
            outputs=list(node.output), name=f"dml_squeeze_ct_{node.name}"
        )

        ct_nodes_to_remove.append(node)
        ct_nodes_to_add.append((i, [weight_reshape_node, unsqueeze_node, ct2d_node, squeeze_node]))
        patched_count += 1
        strides = [a.ints for a in node.attribute if a.name == 'strides']
        print(f"  Patched {node.name}: 1D ConvTranspose -> 2D (kernel={weight_dims[2]}, stride={list(strides[0]) if strides else '?'})")

    for node in ct_nodes_to_remove:
        model.graph.node.remove(node)
    offset = 0
    for idx, new_nodes in sorted(ct_nodes_to_add, key=lambda x: x[0]):
        for j, nn in enumerate(new_nodes):
            model.graph.node.insert(idx + offset + j, nn)
        offset += len(new_nodes) - 1

    # Pass 2: Patch Q/K/V head reshapes and output merges
    # First, scan all Reshape nodes with -1 to auto-detect merge_dim from MatMul weights.
    # This works for any model (talker or vocoder) without needing external config.
    detected_output_dims = []
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue
        has_neg1, known_product, pattern = analyze_reshape_shape(model, node, init_map)
        if not has_neg1:
            continue
        output_dim = trace_to_matmul_weight_dim(model, node.input[0])
        if output_dim is not None:
            detected_output_dims.append(output_dim)

    # The merge_dim is the most common output_dim (hidden_size = num_heads * head_dim)
    auto_merge_dim = None
    if detected_output_dims:
        from collections import Counter
        dim_counts = Counter(detected_output_dims)
        auto_merge_dim = dim_counts.most_common(1)[0][0]

    # Use config-based merge_dim for talker, auto-detected for other models
    if effective_config:
        merge_dim = get_merge_dim_from_config(effective_config) or auto_merge_dim
    else:
        merge_dim = auto_merge_dim

    if merge_dim:
        print(f"  merge_dim={merge_dim} (auto={auto_merge_dim}, config={get_merge_dim_from_config(effective_config) if effective_config else None})")

    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue

        has_neg1, known_product, pattern = analyze_reshape_shape(model, node, init_map)
        if not has_neg1:
            continue

        # Get the data tensor's total output dim from weight tracing
        output_dim = trace_to_matmul_weight_dim(model, node.input[0])

        # Fallback: for output merge patterns [?, ?, -1] where tracing fails,
        # use merge dimension (auto-detected or config-based)
        if output_dim is None and pattern is not None and merge_dim is not None:
            # [?, ?, -1] with known_product=1 -> output merge after attention
            if len(pattern) >= 2 and pattern[-1] == -1 and known_product == 1:
                output_dim = merge_dim

        if output_dim is None:
            skipped_count += 1
            continue

        # Compute the resolved shape
        # For [..., -1, 128]: -1 = output_dim / 128
        # For [..., -1]:      -1 = output_dim
        resolved_shape = []
        resolved_ok = True
        for v in pattern:
            if v == -1:
                if known_product > 0:
                    resolved_val = output_dim // known_product
                    resolved_shape.append(resolved_val)
                else:
                    resolved_ok = False
                    break
            elif v == "?":
                resolved_ok = False
                break
            else:
                resolved_shape.append(v)

        if not resolved_ok:
            # Pattern has dynamic dims — create new Concat with resolved -1
            shape_name = node.input[1]
            concat_node = get_node_producing(model, shape_name)
            if concat_node and concat_node.op_type == "Concat":
                neg1_resolved = output_dim // known_product if known_product > 0 else None
                if neg1_resolved is None:
                    skipped_count += 1
                    continue

                # Build new Concat inputs, replacing -1 constants with resolved values
                new_concat_inputs = []
                for inp in concat_node.input:
                    if inp in init_map and -1 in init_map[inp]:
                        new_name = make_shape_init([neg1_resolved], f"dml_neg1_{node.name}")
                        new_concat_inputs.append(new_name)
                    else:
                        new_concat_inputs.append(inp)

                # Create new Concat node with unique output
                new_concat_output = f"dml_shape_concat_{node.name}"
                new_concat = helper.make_node(
                    "Concat",
                    inputs=new_concat_inputs,
                    outputs=[new_concat_output],
                    axis=0,
                    name=f"dml_concat_{node.name}"
                )

                # Insert the new Concat before the Reshape
                node_idx = list(model.graph.node).index(node)
                model.graph.node.insert(node_idx, new_concat)

                # Point Reshape to new shape
                node.input[1] = new_concat_output
                patched_count += 1
                print(f"  Patched {node.name}: -1 -> {neg1_resolved} (new Concat)")
            else:
                skipped_count += 1
        else:
            # Fully static resolved shape — replace with constant
            new_name = make_shape_init(resolved_shape, f"dml_shape_{node.name}")
            node.input[1] = new_name
            patched_count += 1
            print(f"  Patched {node.name}: {pattern} -> {resolved_shape}")

    print(f"  Total: {patched_count} patched, {skipped_count} skipped")

    if patched_count > 0:
        onnx.save(model, output_path)
        print(f"  Saved to: {output_path}")
    else:
        print("  No changes needed")

    return patched_count


def main():
    parser = argparse.ArgumentParser(description="Patch ONNX models for DirectML compatibility")
    parser.add_argument("model_dir", help="Directory containing ONNX model files")
    parser.add_argument("--output-dir", help="Output directory (default: overwrite in-place)")
    args = parser.parse_args()

    model_files = ["talker_prefill.onnx", "talker_decode.onnx", "code_predictor.onnx", "vocoder.onnx"]
    total_patched = 0

    # Load model config for fallback dimension resolution
    config = load_model_config(args.model_dir)
    if config:
        merge_dim = get_merge_dim_from_config(config)
        print(f"Loaded config: merge_dim={merge_dim}")

    for fname in model_files:
        path = os.path.join(args.model_dir, fname)
        if not os.path.exists(path):
            print(f"Skipping {fname} (not found)")
            continue
        out_path = os.path.join(args.output_dir, fname) if args.output_dir else path
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        total_patched += patch_model(path, out_path, config=config)

    print(f"\nDone. Total patches: {total_patched}")


if __name__ == "__main__":
    main()

