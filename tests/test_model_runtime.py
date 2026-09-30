"""Synthetic ONNX math fixtures test inference contracts, not trained classifiers."""

import json
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from scipy.signal import resample_poly
from scipy.special import expit, softmax

from backend import ecg
from backend.model_runtime import compatibility, predict, validate_model


def _manifest(**changes):
    metadata = {
        "schema_version": 1,
        "name": "合成数学测试模型",
        "version": "test-only",
        "task": "beat_classification",
        "classes": ["second", "first"],
        "sample_rate": 100,
        "input_samples": 64,
        "lead_names": ["II"],
        "pre_samples": 16,
        "normalization": "none",
        "input_filter": "raw",
        "notch": 0,
        "output_type": "logits",
        "threshold": 0.5,
        "description": "合成测试夹具，非训练模型",
    }
    metadata.update(changes)
    return metadata


def _linear_model(
    metadata,
    batch="B",
    rank=2,
    weights=None,
    bias=None,
    input_shape=None,
    output_shape=None,
    dtype=TensorProto.FLOAT,
    nodes=None,
):
    channels, samples, classes = (
        len(metadata["lead_names"]),
        metadata["input_samples"],
        len(metadata["classes"]),
    )
    shape = (
        input_shape
        if input_shape is not None
        else ([batch, samples] if rank == 2 else [batch, channels, samples])
    )
    if weights is None:
        weights = np.zeros((channels * samples, classes), dtype=np.float32)
        weights[0, 0] = 1
        weights[-1, 1] = 1
    if bias is None:
        bias = np.zeros(classes, dtype=np.float32)
    initializers = [
        numpy_helper.from_array(np.asarray(weights, dtype=np.float32), "W"),
        numpy_helper.from_array(np.asarray(bias, dtype=np.float32), "bias"),
    ]
    graph_nodes = (
        nodes
        if nodes is not None
        else [
            helper.make_node("Flatten", ["signal"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "W"], ["weighted"]),
            helper.make_node("Add", ["weighted", "bias"], ["scores"]),
        ]
    )
    graph = helper.make_graph(
        graph_nodes,
        "synthetic-only",
        [helper.make_tensor_value_info("signal", dtype, shape)],
        [
            helper.make_tensor_value_info(
                "scores", TensorProto.FLOAT, output_shape or [batch, classes]
            )
        ],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="PulseLab synthetic test fixture",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 9
    return model


def _validated(metadata, **kwargs):
    return validate_model(_linear_model(metadata, **kwargs).SerializeToString(), metadata)


def _record(original, indices=(100, 200), reliable=True, selected="II"):
    rate = original["sample_rate"]
    return {
        "has_original": True,
        "lead_names": list(original["leads"]),
        "selected_lead": selected,
        "duration": len(next(iter(original["leads"].values()))) / rate,
        "sample_rate": rate,
        "quality": {"reliable": reliable},
        "peaks": [{"index": index, "time": round(index / rate, 6)} for index in indices],
    }


def _original(values=None, rate=100, **leads):
    return {"sample_rate": rate, "units": "mV", "leads": leads or {"II": np.asarray(values)}}


class ModelValidationTests(unittest.TestCase):
    def test_manifest_rejects_ambiguous_or_unsupported_preprocessing(self):
        for changes in (
            {"unknown": 1},
            {"schema_version": True},
            {"classes": ["A", "A"]},
            {"sample_rate": float("nan")},
            {"input_samples": True},
            {"lead_names": ["selected", "II"]},
            {"pre_samples": 64},
            {"input_filter": "raw", "notch": 50},
            {"threshold": 1},
            {"normalization": []},
            {"output_type": "softmax"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                metadata = _manifest(**changes)
                validate_model(b"not-onnx", metadata)

    def test_non_onnx_bytes_fail_with_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "ONNX"):
            validate_model(b"not-a-torch-or-onnx-model", _manifest())

    def test_incompatible_tensor_shapes_and_dtype_are_rejected(self):
        metadata = _manifest()
        for arguments in (
            {"input_shape": [2, 64]},
            {"input_shape": ["B", 65]},
            {"input_shape": ["B", 1, 1, 64]},
            {"dtype": TensorProto.DOUBLE},
            {"rank": 3, "input_shape": ["B", 2, 64]},
            {
                "weights": np.zeros((64, 3), np.float32),
                "bias": np.zeros(3, np.float32),
                "output_shape": ["B", 3],
            },
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                _validated(metadata, **arguments)
        with self.assertRaisesRegex(ValueError, "导联"):
            _validated(_manifest(lead_names=["II", "V1"]), rank=2)

    def test_multiple_inputs_or_outputs_are_rejected(self):
        for additional_input in (True, False):
            model = _linear_model(_manifest())
            if additional_input:
                model.graph.input.append(
                    helper.make_tensor_value_info("extra", TensorProto.FLOAT, ["B", 64])
                )
            else:
                model.graph.output.append(
                    helper.make_tensor_value_info("weighted", TensorProto.FLOAT, ["B", 2])
                )
            with (
                self.subTest(additional_input=additional_input),
                self.assertRaisesRegex(ValueError, "一个输入和一个输出"),
            ):
                validate_model(model.SerializeToString(), _manifest())

    def test_dynamic_export_that_actually_returns_one_row_is_rejected(self):
        metadata = _manifest()
        model = _linear_model(metadata)
        model.graph.node[-1].output[0] = "all_scores"
        model.graph.node.append(
            helper.make_node("ReduceMean", ["all_scores"], ["scores"], axes=[0], keepdims=1)
        )
        with self.assertRaisesRegex(ValueError, "每个输入"):
            validate_model(model.SerializeToString(), metadata)

    def test_external_storage_rejected_in_initializer_constant_subgraph_and_sparse_tensor(self):
        external = helper.make_tensor("external", TensorProto.FLOAT, [1], [0])
        external.data_location = TensorProto.EXTERNAL
        external.external_data.add(key="location", value="never-open-this-file.bin")
        fixtures = []
        model = _linear_model(_manifest())
        model.graph.initializer.append(external)
        fixtures.append(model)
        model = _linear_model(_manifest())
        model.graph.node.append(helper.make_node("Constant", [], ["unused"], value=external))
        fixtures.append(model)
        nested = helper.make_graph(
            [helper.make_node("Constant", [], ["nested"], value=external)],
            "nested",
            [],
            [helper.make_tensor_value_info("nested", TensorProto.FLOAT, [1])],
        )
        model = _linear_model(_manifest())
        model.graph.node.append(
            helper.make_node(
                "If", ["unused_condition"], ["unused"], then_branch=nested, else_branch=nested
            )
        )
        fixtures.append(model)
        model = _linear_model(_manifest())
        sparse = onnx.SparseTensorProto()
        sparse.values.CopyFrom(external)
        sparse.indices.CopyFrom(helper.make_tensor("indices", TensorProto.INT64, [1], [0]))
        sparse.dims.append(1)
        model.graph.sparse_initializer.append(sparse)
        fixtures.append(model)
        for fixture in fixtures:
            with (
                self.subTest(fixture=fixture.graph.node[-1].op_type),
                self.assertRaisesRegex(ValueError, "外部张量"),
            ):
                validate_model(fixture.SerializeToString(), _manifest())

    def test_custom_operator_domains_rejected_before_runtime(self):
        model = _linear_model(_manifest())
        model.graph.node[0].domain = "private.custom"
        model.opset_import.append(helper.make_opsetid("private.custom", 1))
        with self.assertRaisesRegex(ValueError, "算子域"):
            validate_model(model.SerializeToString(), _manifest())

    def test_probabilities_not_renormalized_and_nonfinite_import_rejected(self):
        for probabilities in ([0.2, 0.2], [-0.1, 1.1], [float("nan"), 0.5], [float("inf"), 0.0]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                _validated(
                    _manifest(output_type="probabilities"),
                    weights=np.zeros((64, 2), np.float32),
                    bias=probabilities,
                )


class ModelPredictionTests(unittest.TestCase):
    def test_resampling_uses_original_peak_anchor_preserves_time_and_class_order(self):
        metadata = _manifest(sample_rate=360)
        manifest, session, signature = _validated(metadata)
        raw = np.sin(np.arange(1000) * 0.017) + np.arange(1000) * 0.001
        original = _original(raw, rate=250)
        record = _record(original, indices=(1, 171, 400, 995))
        result = predict(session, manifest, signature, record, original)
        transformed = resample_poly(raw, 36, 25)
        self.assertEqual(result["summary"]["skipped"], 2)
        self.assertEqual(result["preprocessing"]["resample_up"], 36)
        self.assertEqual(result["preprocessing"]["resample_down"], 25)
        for item, peak in zip(result["predictions"], record["peaks"][1:3], strict=True):
            offset = round(peak["index"] * 360 / 250) - 16
            expected = softmax(
                np.asarray(
                    [transformed[offset], transformed[offset + 63]], dtype=np.float32
                ).astype(np.float64)
            )
            np.testing.assert_allclose(item["scores"], expected, atol=1e-7)
            self.assertEqual(item["index"], peak["index"])
            self.assertEqual(item["time"], peak["time"])
            self.assertEqual(item["start"], offset / 360)
            self.assertEqual(item["end"], (offset + 64) / 360)
            self.assertEqual(item["class_label"], metadata["classes"][int(expected.argmax())])
        self.assertTrue(result["warnings"])
        json.dumps(result, allow_nan=False)

    def test_dynamic_and_fixed_batch_produce_same_all_beats_and_counts_include_zero(self):
        metadata = _manifest()
        original = _original(np.linspace(2, -2, 5000))
        record = _record(original, indices=tuple(range(100, 4400, 100)))
        results = []
        for batch in ("B", 1):
            manifest, session, signature = _validated(metadata, batch=batch)
            results.append(predict(session, manifest, signature, record, original))
        self.assertEqual(results[0]["predictions"], results[1]["predictions"])
        self.assertEqual(
            results[0]["summary"],
            {
                "evaluated": 43,
                "skipped": 0,
                "counts": [{"label": "second", "count": 43}, {"label": "first", "count": 0}],
            },
        )

    def test_ordered_leads_and_selected_analysis_lead_not_display_waveform(self):
        metadata = _manifest(
            task="record_classification", pre_samples=None, lead_names=["V1", "II"]
        )
        weights = np.zeros((128, 2), np.float32)
        weights[0, 0], weights[64, 1] = 1, 1
        manifest, session, signature = _validated(metadata, rank=3, weights=weights)
        original = _original(II=np.full(1000, -1.0), V1=np.full(1000, 2.0))
        record = _record(original, selected="II")
        record["waveform"] = {"raw": [9999], "leads": {"V1": [-9999]}}
        result = predict(session, manifest, signature, record, original)
        self.assertEqual(result["input_leads"], ["V1", "II"])
        np.testing.assert_allclose(result["predictions"][0]["scores"], softmax([2.0, -1.0]))
        manifest, session, signature = _validated(
            _manifest(task="record_classification", pre_samples=None, lead_names=["selected"])
        )
        result = predict(session, manifest, signature, record, original)
        self.assertEqual(result["input_leads"], ["II"])
        np.testing.assert_allclose(result["predictions"][0]["scores"], [0.5, 0.5])

    def test_filtered_resampled_predictions_keep_full_source_context(self):
        metadata = _manifest(sample_rate=360, input_filter="bandpass", notch=50)
        manifest, session, signature = _validated(metadata)
        original = _original(np.sin(np.arange(2000) * 0.1), rate=250)
        record = _record(original, indices=tuple(range(200, 1800, 100)))
        expected = resample_poly(ecg._filter_waveform(original["leads"]["II"], 250, 50), 36, 25)
        result = predict(session, manifest, signature, record, original)
        first = round(200 * 360 / 250) - 16
        np.testing.assert_allclose(
            result["predictions"][0]["scores"],
            softmax(np.asarray([expected[first], expected[first + 63]], np.float32)),
            atol=1e-7,
        )

    def test_requested_notch_cannot_be_silently_skipped_at_source_nyquist(self):
        manifest, session, signature = _validated(
            _manifest(sample_rate=360, input_filter="bandpass", notch=50)
        )
        original = _original(np.sin(np.arange(1000) * 0.1), rate=100)
        record = _record(original)
        with self.assertRaises(ValueError):
            predict(session, manifest, signature, record, original)
        self.assertFalse(compatibility(manifest, record)[0])

    def test_per_lead_per_window_normalization_including_constant_lead(self):
        original = _original(II=np.arange(1000, dtype=float), V1=np.full(1000, 9.0))
        record = _record(original, indices=(100, 300))
        weights = np.zeros((128, 2), np.float32)
        weights[0, 0], weights[64, 1] = 1, 1
        for method in ("zscore", "minmax"):
            manifest, session, signature = _validated(
                _manifest(lead_names=["II", "V1"], normalization=method), rank=3, weights=weights
            )
            result = predict(session, manifest, signature, record, original)
            segment = np.arange(64, dtype=float)
            expected_first = (
                (segment[0] - segment.mean()) / segment.std() if method == "zscore" else 0
            )
            for item in result["predictions"]:
                np.testing.assert_allclose(item["scores"], softmax([expected_first, 0]), atol=1e-7)

    def test_record_window_rounding_exact_end_and_rejected_partial_window(self):
        manifest, session, signature = _validated(
            _manifest(task="record_classification", pre_samples=None)
        )
        original = _original(np.arange(1000, dtype=float) * 0.001)
        record = _record(original, reliable=False)
        result = predict(session, manifest, signature, record, original, start=9.36)
        item = result["predictions"][0]
        self.assertIsNone(item["index"])
        self.assertEqual((item["time"], item["start"], item["end"]), (9.36, 9.36, 10.0))
        rounded = predict(session, manifest, signature, record, original, start=0.014)
        self.assertEqual(rounded["predictions"][0]["start"], 0.01)
        with self.assertRaisesRegex(ValueError, "完整模型窗口"):
            predict(session, manifest, signature, record, original, start=9.37)
        for start in (-1, float("nan"), float("inf")):
            with self.subTest(start=start), self.assertRaises(ValueError):
                predict(session, manifest, signature, record, original, start=start)

    def test_extreme_single_label_logits_are_stable_without_clipping(self):
        metadata = _manifest(task="record_classification", pre_samples=None)
        manifest, session, signature = _validated(
            metadata, weights=np.zeros((64, 2), np.float32), bias=[1000, 999]
        )
        original = _original(np.zeros(1000))
        result = predict(session, manifest, signature, _record(original), original)
        np.testing.assert_allclose(
            result["predictions"][0]["scores"], softmax([0.0, -1.0]), atol=1e-12
        )
        json.dumps(result, allow_nan=False)

    def test_resampling_ceil_length_does_not_invent_complete_source_window(self):
        manifest, session, signature = _validated(_manifest(sample_rate=360))
        original = _original(np.zeros(1001), rate=250)
        result = predict(
            session, manifest, signature, _record(original, indices=(100, 968)), original
        )
        self.assertEqual([item["index"] for item in result["predictions"]], [100])
        self.assertEqual(result["summary"]["skipped"], 1)

    def test_multilabel_sigmoid_threshold_order_and_no_labels(self):
        metadata = _manifest(task="record_multilabel", pre_samples=None, classes=["C", "A", "B"])
        original = _original(np.zeros(1000))
        record = _record(original, reliable=False)
        for threshold in (0.5, 0.999):
            metadata["threshold"] = threshold
            manifest, session, signature = _validated(
                metadata, weights=np.zeros((64, 3), np.float32), bias=[0, -1000, 2]
            )
            result = predict(session, manifest, signature, record, original)
            item = result["predictions"][0]
            np.testing.assert_allclose(item["scores"], expit([0, -1000, 2]))
            self.assertIsNone(item["class_index"])
            self.assertEqual(item["labels"], ["C", "B"] if threshold == 0.5 else [])
            self.assertEqual(item["class_label"], "C、B" if threshold == 0.5 else "未超过阈值")
            self.assertEqual(
                result["summary"]["counts"],
                [
                    {"label": "C", "count": int(threshold == 0.5)},
                    {"label": "A", "count": 0},
                    {"label": "B", "count": int(threshold == 0.5)},
                ],
            )
            json.dumps(result, allow_nan=False)

    def test_valid_probability_scores_preserved_without_softmax(self):
        metadata = _manifest(
            task="record_classification", pre_samples=None, output_type="probabilities"
        )
        manifest, session, signature = _validated(
            metadata, weights=np.zeros((64, 2), np.float32), bias=[0.8, 0.2]
        )
        original = _original(np.zeros(1000))
        result = predict(session, manifest, signature, _record(original), original)
        np.testing.assert_allclose(result["predictions"][0]["scores"], [0.8, 0.2], atol=1e-7)

    def test_runtime_rejects_data_dependent_invalid_probability_and_nonfinite_output(self):
        metadata = _manifest(
            task="record_classification", pre_samples=None, output_type="probabilities"
        )
        weights = np.zeros((64, 2), np.float32)
        weights[0, 0] = 1
        manifest, session, signature = _validated(metadata, weights=weights, bias=[0.5, 0.5])
        original = _original(np.ones(1000))
        with self.assertRaisesRegex(ValueError, "0–1"):
            predict(session, manifest, signature, _record(original), original)
        metadata = _manifest(task="record_classification", pre_samples=None)
        nodes = [
            helper.make_node("Flatten", ["signal"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "W"], ["weighted"]),
            helper.make_node("Exp", ["weighted"], ["scores"]),
        ]
        manifest, session, signature = _validated(metadata, nodes=nodes)
        original = _original(np.full(1000, 1000.0))
        with self.assertRaisesRegex(ValueError, "非|有限"):
            predict(session, manifest, signature, _record(original), original)

    def test_reliability_original_leads_shortness_and_all_edge_beats_are_actionable(self):
        metadata = _manifest()
        manifest, session, signature = _validated(metadata)
        original = _original(np.zeros(1000))
        record = _record(original)
        for key, value, reason in (
            ("has_original", False, "原始"),
            ("lead_names", ["V1"], "缺少"),
            ("quality", {"reliable": False}, "不可靠"),
            ("peaks", [], "R 峰"),
        ):
            changed = {**record, key: value}
            self.assertFalse(compatibility(manifest, changed)[0])
            with self.assertRaisesRegex(ValueError, reason):
                predict(session, manifest, signature, changed, original)
        with self.assertRaisesRegex(ValueError, "所有 R 峰"):
            predict(session, manifest, signature, _record(original, indices=(1, 999)), original)
        with self.assertRaisesRegex(ValueError, "索引"):
            predict(session, manifest, signature, _record(original, indices=(1000,)), original)
        record_manifest = _manifest(task="record_classification", pre_samples=None)
        short = _original(np.zeros(63))
        self.assertFalse(compatibility(record_manifest, _record(short, indices=()))[0])
        corrupted = _original(np.full(1000, float("nan")))
        with self.assertRaisesRegex(ValueError, "有限"):
            predict(session, manifest, signature, record, corrupted)


if __name__ == "__main__":
    unittest.main()
