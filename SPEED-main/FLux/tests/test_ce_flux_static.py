import ast
import pathlib


CE_FLUX_PATH = pathlib.Path(__file__).resolve().parents[1] / "CE_Flux.py"


def _source():
    return CE_FLUX_PATH.read_text()


def test_cli_exposes_attention_params_with_kv_default():
    tree = ast.parse(_source())
    add_argument_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ]

    params_calls = [
        call
        for call in add_argument_calls
        if call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "--params"
    ]

    assert params_calls, "--params should select K/V/QKV attention edit targets"
    params_call = params_calls[0]
    defaults = {
        keyword.arg: keyword.value.value
        for keyword in params_call.keywords
        if isinstance(keyword.value, ast.Constant)
    }
    assert defaults["default"] == "KV"


def test_text_side_value_projection_is_editable():
    source = _source()
    assert ".attn.add_v_proj" in source
    assert ".attn.add_k_proj" in source
    assert "args.params" in source


def test_anchor_trace_is_recomputed_inside_layer_update_loop():
    source = _source()
    layer_update_source = source[source.index("# region [Layer Update]"):]
    loop_source = layer_update_source[layer_update_source.index("for module_index"):]

    anchor_pos = loop_source.index("anchor_traces = _trace_many(pipeline, anchor_concepts")
    target_mean_pos = loop_source.index("target_mean = _mean_outputs")
    edit_trace_pos = loop_source.index("edit_traces = _trace_many(pipeline, target_concepts")

    assert anchor_pos < target_mean_pos < edit_trace_pos
