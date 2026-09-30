"""Deterministic labeled waveforms exercise import and observable DSP behavior."""

import json
import unittest

import numpy as np

from backend.ecg import _beat_template, _filter_waveform, _rr_tools, analyze_signal, parse_upload


def _ecg(rate=250, duration=12, polarity=1, t_amplitude=0.3, beats=None):
    time = np.arange(int(rate * duration)) / rate
    if beats is None:
        beats = np.arange(0.7, duration - 0.3, 0.8)
    waveform = 0.025 * np.sin(2 * np.pi * 0.27 * time)
    for beat in beats:
        waveform += 0.10 * np.exp(-0.5 * ((time - beat + 0.18) / 0.035) ** 2)
        waveform -= 0.12 * np.exp(-0.5 * ((time - beat + 0.026) / 0.010) ** 2)
        waveform += 1.00 * np.exp(-0.5 * ((time - beat) / 0.014) ** 2)
        waveform -= 0.20 * np.exp(-0.5 * ((time - beat - 0.035) / 0.014) ** 2)
        waveform += t_amplitude * np.exp(-0.5 * ((time - beat - 0.27) / 0.090) ** 2)
    return polarity * waveform, np.asarray(beats)


def _upload(payload, rate=250, units="mV"):
    return parse_upload(json.dumps(payload).encode(), "record.json", rate, units)


class ECGImportTests(unittest.TestCase):
    def test_time_column_calibrates_rate_and_microvolts(self):
        rate = 250
        waveform, _ = _ecg(rate=rate, duration=5)
        lines = ["time_s;II;V1"]
        for index, value in enumerate(waveform):
            lines.append(f"{13 + index / rate:.6f};{value * 1000:.9e};{-value * 500:.9e}")
        parsed = parse_upload("\n".join(lines).encode(), "capture.csv", 500, "uV")
        self.assertAlmostEqual(parsed["sample_rate"], rate, places=6)
        self.assertEqual(parsed["units"], "mV")
        np.testing.assert_allclose(parsed["leads"]["II"], waveform, atol=1e-9)
        np.testing.assert_allclose(parsed["leads"]["V1"], -waveform / 2, atol=1e-9)
        self.assertTrue(parsed["warnings"])

    def test_rounded_timestamps_accept_five_second_boundary(self):
        waveform, _ = _ecg(rate=360, duration=5)
        rows = ["time_s,II"] + [
            f"{index / 360:.8f},{value:.6f}" for index, value in enumerate(waveform)
        ]
        parsed = parse_upload("\n".join(rows).encode(), "rounded.csv", 250, "mV")
        result = analyze_signal(parsed["leads"], parsed["sample_rate"], "II")
        self.assertAlmostEqual(result["duration"], 5.0, places=5)
        self.assertAlmostEqual(result["metrics"]["heart_rate"], 75.0, delta=1.0)

    def test_scientific_notation_without_header_keeps_first_sample(self):
        values = [(1e-3 + index * 1e-6, -2e-3) for index in range(500)]
        text = "\n".join(f"{a:.9e}\t{b:.9e}" for a, b in values)
        parsed = parse_upload(text.encode(), "numeric.txt", 100, "V")
        self.assertEqual(len(parsed["leads"]), 2)
        first, second = parsed["leads"].values()
        self.assertEqual(len(first), 500)
        self.assertAlmostEqual(first[0], 1.0)
        self.assertAlmostEqual(first[-1], values[-1][0] * 1000)
        self.assertAlmostEqual(second[0], -2.0)

    def test_nonuniform_and_nonmonotonic_time_are_rejected(self):
        time = np.arange(500) / 100
        time[240:] += 0.005
        text = "time,II\n" + "\n".join(f"{t:.6f},0.1" for t in time)
        with self.assertRaisesRegex(ValueError, "不均匀"):
            parse_upload(text.encode(), "irregular.csv", 100, "mV")
        time[240] = time[239]
        text = "timestamp,II\n" + "\n".join(f"{t:.6f},0.1" for t in time)
        with self.assertRaisesRegex(ValueError, "严格递增"):
            parse_upload(text.encode(), "duplicate.csv", 100, "mV")

    def test_missing_or_nonnumeric_cells_never_drop_rows(self):
        valid = ["0.1,0.2"] * 500
        broken = valid.copy()
        broken[127] = "0.1,"
        with self.assertRaisesRegex(ValueError, "缺少数值"):
            parse_upload("\n".join(broken).encode(), "missing.csv", 100, "mV")
        broken[127] = "0.1"
        with self.assertRaisesRegex(ValueError, "列数不一致"):
            parse_upload("\n".join(broken).encode(), "ragged.csv", 100, "mV")
        broken[127] = "0.1,坏值"
        with self.assertRaisesRegex(ValueError, "不是有效数字"):
            parse_upload("\n".join(broken).encode(), "corrupt.csv", 100, "mV")
        with self.assertRaisesRegex(ValueError, "首行混有"):
            parse_upload(("1e-3,坏值\n" + "\n".join(valid)).encode(), "first.csv", 100, "mV")

    def test_json_rejects_nonfinite_missing_and_duplicate_leads(self):
        waveform = [0.1] * 1250
        waveform[400] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN"):
            _upload({"signal": waveform})
        waveform[400] = None
        with self.assertRaisesRegex(ValueError, "空值"):
            _upload({"signal": waveform})
        duplicate = '{"leads":{"II":[0],"II":[1]}}'
        with self.assertRaisesRegex(ValueError, "重复"):
            parse_upload(duplicate.encode(), "duplicate.json", 250, "mV")
        with self.assertRaisesRegex(ValueError, "相同的采样点数"):
            _upload({"leads": {"II": [0.1] * 1250, "V1": [0.1] * 1249}})

    def test_lead_duration_rate_and_magnitude_boundaries(self):
        leads = {f"L{i}": [0.1] * 500 for i in range(12)}
        parsed = _upload({"sample_rate": 100, "leads": leads})
        self.assertEqual(list(parsed["leads"]), list(leads))
        leads["L12"] = [0.1] * 500
        with self.assertRaisesRegex(ValueError, "12"):
            _upload({"sample_rate": 100, "leads": leads})
        with self.assertRaisesRegex(ValueError, "5–300"):
            _upload({"sample_rate": 100, "signal": [0.1] * 499})
        with self.assertRaisesRegex(ValueError, "100–1000"):
            _upload({"sample_rate": 99, "signal": [0.1] * 500})
        with self.assertRaisesRegex(ValueError, "幅值"):
            _upload({"sample_rate": 100, "units": "V", "signal": [0.101] * 500})
        with self.assertRaisesRegex(ValueError, "300000"):
            _upload({"sample_rate": 1000, "signal": [0.1] * 300001})


class ECGAnalysisTests(unittest.TestCase):
    def test_constant_signal_never_publishes_heart_rate(self):
        for constant in (0.0, 0.25):
            result = analyze_signal({"II": [constant] * 1250}, 250, "II")
            self.assertFalse(result["quality"]["reliable"])
            self.assertEqual(result["peaks"], [])
            self.assertEqual(result["metrics"]["beat_count"], 0)
            for key, value in result["metrics"].items():
                if key != "beat_count":
                    self.assertIsNone(value)
            json.dumps(result, allow_nan=False)
        waveform, _ = _ecg()
        waveform[500] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            analyze_signal({"II": waveform.tolist()}, 250, "II")

    def test_inverted_qrs_matches_labeled_r_peaks(self):
        waveform, beats = _ecg(polarity=-1)
        result = analyze_signal({"aVR": waveform.tolist()}, 250, "aVR", notch=0)
        measured = np.asarray([peak["time"] for peak in result["peaks"]])
        self.assertEqual(len(measured), len(beats))
        np.testing.assert_allclose(measured, beats, atol=0.024)
        self.assertTrue(result["quality"]["reliable"])
        self.assertAlmostEqual(result["metrics"]["heart_rate"], 75.0, delta=1.0)
        self.assertAlmostEqual(result["metrics"]["mean_rr"], 800.0, delta=8.0)
        self.assertTrue(all(peak["amplitude"] < 0 for peak in result["peaks"]))
        json.dumps(result, allow_nan=False)

    def test_larger_t_waves_are_not_counted_as_r_peaks(self):
        waveform, beats = _ecg(t_amplitude=2.5)
        # T is genuinely taller than R in the input, not just a nominal parameter.
        self.assertGreater(waveform[int((beats[0] + 0.27) * 250)], waveform[int(beats[0] * 250)])
        result = analyze_signal({"II": waveform.tolist()}, 250, "II", notch=0)
        measured = np.asarray([peak["time"] for peak in result["peaks"]])
        self.assertEqual(len(measured), len(beats))
        np.testing.assert_allclose(measured, beats, atol=0.024)
        self.assertAlmostEqual(result["metrics"]["heart_rate"], 75.0, delta=1.0)
        self.assertTrue(
            np.all(np.min(np.abs(measured[:, None] - (beats + 0.27)[None, :]), axis=1) > 0.15)
        )

    def test_broadband_noise_withholds_hr_statistics(self):
        noise = np.random.default_rng(1729).normal(0, 0.2, 3000)
        result = analyze_signal({"II": noise.tolist()}, 250, "II")
        self.assertFalse(result["quality"]["reliable"])
        self.assertIsNone(result["metrics"]["heart_rate"])
        self.assertIsNone(result["metrics"]["mean_rr"])
        self.assertIsNone(result["metrics"]["rmssd"])
        json.dumps(result, allow_nan=False)

    def test_rr_statistics_follow_measured_irregular_intervals(self):
        beats = np.concatenate(
            (
                [0.7],
                0.7 + np.cumsum([0.6, 1.0, 0.7, 0.95, 0.65, 1.0, 0.7, 0.95, 0.65, 1.0, 0.7, 0.95]),
            )
        )
        waveform, _ = _ecg(beats=beats)
        result = analyze_signal({"II": waveform.tolist()}, 250, "II", notch=0)
        measured = np.asarray([peak["time"] for peak in result["peaks"]])
        np.testing.assert_allclose(measured, beats, atol=0.024)
        rr = np.diff(measured) * 1000
        self.assertAlmostEqual(result["metrics"]["sdnn"], float(np.std(rr, ddof=1)), delta=0.02)
        self.assertAlmostEqual(
            result["metrics"]["rmssd"],
            float(np.sqrt(np.mean(np.diff(rr) ** 2))),
            delta=0.02,
        )
        self.assertEqual(result["metrics"]["pnn50"], 100.0)
        self.assertTrue(any(row["flag"] for row in result["rr"]))

    def test_downsampling_preserves_all_lead_extrema_on_shared_time_grid(self):
        rate = 250
        length = 15000
        waveform, _ = _ecg(rate=rate, duration=length / rate)
        leads = {}
        spike_indices = []
        for lead in range(12):
            values = waveform.copy() * (0.2 + lead / 20)
            positive, negative = 311 + lead * 1013, 349 + lead * 1013
            values[positive], values[negative] = 8 + lead / 10, -8 - lead / 10
            spike_indices.extend((positive, negative))
            leads[f"L{lead}"] = values.tolist()
        result = analyze_signal(leads, rate, "L0", notch=0)
        display = result["waveform"]
        indices = np.rint(np.asarray(display["time"]) * rate).astype(int)
        self.assertLessEqual(len(indices), 12000)
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], length - 1)
        self.assertTrue(set(spike_indices).issubset(set(indices.tolist())))
        for name, original in leads.items():
            np.testing.assert_array_equal(display["leads"][name], np.asarray(original)[indices])
        np.testing.assert_array_equal(display["raw"], display["leads"]["L0"])
        self.assertEqual(len(display["filtered"]), len(indices))

    def test_templates_use_original_samples_and_cap_statistical_and_preview_beats(self):
        rate = 1000
        waveform, _ = _ecg(rate=rate, duration=180)
        result = analyze_signal({"II": waveform.tolist()}, rate, "II", notch=0)
        template = result["tools"]["template"]
        self.assertTrue(template["reliable"])
        self.assertGreater(template["available_count"], 200)
        self.assertEqual(template["used_count"], 200)
        self.assertEqual(len(template["traces"]), 30)
        self.assertEqual(len(template["peak_times"]), 30)
        np.testing.assert_array_equal(template["time_ms"], np.linspace(-200, 400, 201))
        full_filtered = _filter_waveform(waveform, rate, 0)
        source_positions = np.arange(len(waveform))
        offsets = np.asarray(template["time_ms"]) / 1000
        displayed = result["waveform"]
        largest_display_error = 0.0
        for trace, peak_time in zip(template["traces"], template["peak_times"], strict=True):
            times = peak_time + offsets
            original = np.interp(times * rate, source_positions, full_filtered)
            # Baseline removal may add a constant, never change the waveform shape.
            np.testing.assert_allclose(np.diff(trace), np.diff(original), atol=1e-10)
            decimated = np.interp(times, displayed["time"], displayed["filtered"])
            largest_display_error = max(
                largest_display_error, float(np.max(np.abs(np.diff(original) - np.diff(decimated))))
            )
        self.assertGreater(largest_display_error, 0.001)
        median, p25, p75 = (np.asarray(template[key]) for key in ("median", "p25", "p75"))
        self.assertTrue(np.all(p25 <= median))
        self.assertTrue(np.all(median <= p75))
        spectrum = result["tools"]["spectrum"]
        self.assertLessEqual(len(spectrum["frequency_hz"]), 512)
        self.assertEqual(len(spectrum["frequency_hz"]), len(spectrum["raw_power"]))
        self.assertEqual(len(spectrum["raw_power"]), len(spectrum["filtered_power"]))
        self.assertLessEqual(max(spectrum["frequency_hz"]), 100)
        self.assertTrue(np.all(np.asarray(spectrum["raw_power"]) >= 0))
        self.assertTrue(np.all(np.asarray(spectrum["filtered_power"]) >= 0))
        json.dumps(result, allow_nan=False)

    def test_template_boundary_windows_are_never_padded_or_clipped(self):
        rate = 100
        source = np.sin(np.arange(500) / 9)
        indices = np.asarray([19, 20, 200, 459, 460])
        template = _beat_template(source, rate, indices, True)
        self.assertEqual(template["available_count"], 3)
        self.assertEqual(template["used_count"], 3)
        np.testing.assert_array_equal(template["peak_times"], [0.2, 2.0, 4.59])
        for trace, peak in zip(template["traces"], (20, 200, 459), strict=True):
            positions = peak + np.asarray(template["time_ms"]) * rate / 1000
            expected = np.interp(positions, np.arange(len(source)), source)
            np.testing.assert_allclose(np.diff(trace), np.diff(expected), atol=1e-12)
        insufficient = _beat_template(source, rate, np.asarray([19, 20, 459, 460]), True)
        self.assertEqual(insufficient["available_count"], 2)
        self.assertEqual(insufficient["used_count"], 0)
        self.assertFalse(insufficient["reliable"])
        self.assertTrue(insufficient["reason"])
        for key in ("time_ms", "median", "p25", "p75", "traces", "peak_times"):
            self.assertEqual(insufficient[key], [])

    def test_templates_preserve_inverted_morphology_and_millivolt_scale(self):
        waveform, _ = _ecg()
        positive = analyze_signal({"II": waveform.tolist()}, 250, "II", notch=0)
        negative = analyze_signal({"aVR": (-2 * waveform).tolist()}, 250, "aVR", notch=0)
        template = negative["tools"]["template"]
        reference = positive["tools"]["template"]
        self.assertTrue(template["reliable"])
        np.testing.assert_allclose(template["peak_times"], reference["peak_times"], atol=1e-9)
        np.testing.assert_allclose(
            template["median"], -2 * np.asarray(reference["median"]), atol=1e-10
        )
        np.testing.assert_allclose(template["p25"], -2 * np.asarray(reference["p75"]), atol=1e-10)
        np.testing.assert_allclose(template["p75"], -2 * np.asarray(reference["p25"]), atol=1e-10)
        center = int(np.argmin(np.abs(template["time_ms"])))
        self.assertLess(template["median"][center], -1)

    def test_unreliable_signals_withhold_templates_and_rr_tools_but_keep_real_psd(self):
        for waveform in (np.zeros(3000), np.random.default_rng(1729).normal(0, 0.2, 3000)):
            result = analyze_signal({"II": waveform.tolist()}, 250, "II")
            self.assertFalse(result["quality"]["reliable"])
            tools = result["tools"]
            template = tools["template"]
            self.assertFalse(template["reliable"])
            self.assertEqual(template["used_count"], 0)
            self.assertTrue(template["reason"])
            for key in ("time_ms", "median", "p25", "p75", "traces", "peak_times"):
                self.assertEqual(template[key], [])
            self.assertEqual(tools["rr_histogram"], {"edges_ms": [], "counts": []})
            self.assertEqual(tools["poincare"], {"x_ms": [], "y_ms": []})
            if not np.any(waveform):
                np.testing.assert_array_equal(tools["spectrum"]["raw_power"], 0)
                np.testing.assert_array_equal(tools["spectrum"]["filtered_power"], 0)
            else:
                self.assertGreater(float(np.sum(tools["spectrum"]["raw_power"])), 0)
            json.dumps(result, allow_nan=False)

    def test_rr_poincare_never_bridges_excluded_intervals(self):
        intervals = np.asarray([0.8, 0.9, 2.4, 0.7, 1.0, 0.2, 0.6, 0.85])
        plausible = (intervals >= 0.3) & (intervals <= 2.0)
        histogram, poincare = _rr_tools(intervals, plausible, True)
        np.testing.assert_allclose(poincare["x_ms"], [800, 700, 600])
        np.testing.assert_allclose(poincare["y_ms"], [900, 1000, 850])
        self.assertEqual(sum(histogram["counts"]), 6)
        self.assertEqual(len(histogram["counts"]), 10)
        np.testing.assert_allclose(np.diff(histogram["edges_ms"]), 40)
        constant, pairs = _rr_tools(np.full(4, 0.8), np.ones(4, dtype=bool), True)
        self.assertEqual(sum(constant["counts"]), 4)
        self.assertLess(constant["edges_ms"][0], 800)
        self.assertGreater(constant["edges_ms"][-1], 800)
        np.testing.assert_allclose(pairs["x_ms"], [800, 800, 800])
        np.testing.assert_allclose(pairs["y_ms"], [800, 800, 800])
        waveform, _ = _ecg()
        result = analyze_signal({"II": waveform.tolist()}, 250, "II", notch=0)
        self.assertEqual(
            [row["peak_index"] for row in result["rr"]],
            [peak["index"] for peak in result["peaks"][1:]],
        )


if __name__ == "__main__":
    unittest.main()
