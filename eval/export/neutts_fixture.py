# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy>=2", "onnx>=1.17", "onnxruntime>=1.22"]
# ///
"""Tiny stand-ins with NeuTTS's exact graph IO, so the Rust loop (prompt, cache, windows, overlap-add) runs
in ordinary tests without the gated weights. The backbone writes speech code 7 until the sequence holds
`--end` tokens, then END; the decoder voices 480 samples of 0.1 per code; the encoder emits one zero
code per 320 samples. Run once: `uv run --script eval/export/neutts_fixture.py --out evoice/tts/tests/resources/neutts`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper, numpy_helper

LAYERS, HEADS, DIM = 24, 3, 64
END, SPEECH, VOCAB = 128_261, 128_262, 193_798
OPSET = [helper.make_opsetid("", 17)]


def constant(name: str, value: np.ndarray) -> onnx.NodeProto:
    return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(value, name))


def backbone(end: int) -> onnx.ModelProto:
    nodes = [
        constant("one", np.array([1], np.int64)),
        constant("second", np.array([2], np.int64)),
        constant("zero_row", np.zeros((1, HEADS, 1, DIM), np.float32)),
        constant("heads", np.array([1, HEADS], np.int64)),
        constant("dim", np.array([DIM], np.int64)),
        constant("vocab", np.array(VOCAB, np.int64)),
        constant("start", np.array(0, np.int64)),
        constant("step", np.array(1, np.int64)),
        constant("limit", np.array(end, np.int64)),
        constant("end_id", np.array(END, np.int64)),
        constant("code_id", np.array(SPEECH + 7, np.int64)),
        constant("peak", np.array(20.0, np.float32)),
        helper.make_node("Shape", ["input_ids"], ["ids_shape"]),
        helper.make_node("Slice", ["ids_shape", "one", "second"], ["s"]),
        helper.make_node("Shape", ["attention_mask"], ["mask_shape"]),
        helper.make_node("Slice", ["mask_shape", "one", "second"], ["t"]),
        helper.make_node("Squeeze", ["t"], ["t_scalar"]),
        helper.make_node("Concat", ["heads", "s", "dim"], ["row_shape"], axis=0),
        helper.make_node("Expand", ["zero_row", "row_shape"], ["fresh"]),
        helper.make_node("Greater", ["t_scalar", "limit"], ["over"]),
        helper.make_node("Where", ["over", "end_id", "code_id"], ["target"]),
        helper.make_node("Range", ["start", "vocab", "step"], ["positions"]),
        helper.make_node("Equal", ["positions", "target"], ["hot"]),
        helper.make_node("Cast", ["hot"], ["hot_f"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["hot_f", "peak"], ["row"]),
        helper.make_node("Unsqueeze", ["row", "start_axes"], ["row3"]),
        helper.make_node("Concat", ["one", "s", "vocab_1d"], ["logit_shape"], axis=0),
        helper.make_node("Expand", ["row3", "logit_shape"], ["logits"]),
    ]
    nodes.insert(0, constant("start_axes", np.array([0, 1], np.int64)))
    nodes.insert(0, constant("vocab_1d", np.array([VOCAB], np.int64)))
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, "s"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, "t"]),
    ]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "s", VOCAB])]
    for layer in range(LAYERS):
        for kind in ("key", "value"):
            past, present = f"past_key_values.{layer}.{kind}", f"present.{layer}.{kind}"
            inputs.append(helper.make_tensor_value_info(past, TensorProto.FLOAT, [1, HEADS, "p", DIM]))
            outputs.append(helper.make_tensor_value_info(present, TensorProto.FLOAT, [1, HEADS, "q", DIM]))
            nodes.append(helper.make_node("Concat", [past, "fresh"], [present], axis=2))
    graph = helper.make_graph(nodes, "backbone", inputs, outputs)
    return helper.make_model(graph, opset_imports=OPSET, ir_version=9)


def decoder() -> onnx.ModelProto:
    nodes = [
        constant("value", np.full((1, 1, 1), 0.1, np.float32)),
        constant("hop", np.array([480], np.int64)),
        constant("two", np.array([2], np.int64)),
        constant("three", np.array([3], np.int64)),
        constant("head", np.array([1, 1], np.int64)),
        helper.make_node("Shape", ["codes"], ["shape"]),
        helper.make_node("Slice", ["shape", "two", "three"], ["frames"]),
        helper.make_node("Sub", ["frames", "one"], ["less"]),
        helper.make_node("Mul", ["less", "hop"], ["samples"]),
        helper.make_node("Concat", ["head", "samples"], ["out_shape"], axis=0),
        helper.make_node("Expand", ["value", "out_shape"], ["audio"]),
    ]
    nodes.insert(0, constant("one", np.array([1], np.int64)))
    graph = helper.make_graph(
        nodes,
        "decoder",
        [helper.make_tensor_value_info("codes", TensorProto.INT32, [1, 1, "f"])],
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, "n"])],
    )
    return helper.make_model(graph, opset_imports=OPSET, ir_version=9)


def encoder() -> onnx.ModelProto:
    nodes = [
        constant("zero", np.zeros((1, 1, 1), np.int32)),
        constant("hop", np.array([320], np.int64)),
        constant("two", np.array([2], np.int64)),
        constant("three", np.array([3], np.int64)),
        constant("head", np.array([1, 1], np.int64)),
        helper.make_node("Shape", ["audio"], ["shape"]),
        helper.make_node("Slice", ["shape", "two", "three"], ["samples"]),
        helper.make_node("Div", ["samples", "hop"], ["frames"]),
        helper.make_node("Concat", ["head", "frames"], ["out_shape"], axis=0),
        helper.make_node("Expand", ["zero", "out_shape"], ["codes"]),
    ]
    graph = helper.make_graph(
        nodes,
        "encoder",
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, "t"])],
        [helper.make_tensor_value_info("codes", TensorProto.INT32, [1, 1, "f"])],
    )
    return helper.make_model(graph, opset_imports=OPSET, ir_version=9)


def tokenizer() -> dict:
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--end", type=int, default=1000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, model in (("model", backbone(args.end)), ("decoder", decoder()), ("distill_neucodec_encoder", encoder())):
        onnx.checker.check_model(model)
        onnx.save(model, args.out / f"{name}.onnx")
        onnxruntime.InferenceSession(args.out / f"{name}.onnx")
    (args.out / "tokenizer.json").write_text(json.dumps(tokenizer()), encoding="utf-8")
    print(f"fixtures in {args.out}")


if __name__ == "__main__":
    main()
