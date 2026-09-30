import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("health", Path(__file__).resolve().parents[1] /
                                           "scripts/infra/check_a800_gpu_health.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def healthy():
    return [dict(uuid=f"GPU-{i}", name="NVIDIA A800 80GB PCIe", recovery="None",
        ecc_mode="Enabled", remap_pending="No", remap_failure="No", pids=[],
        counters={"volatile/dram_uncorrectable": 0, "aggregate/dram_uncorrectable": 71,
                  "volatile/dram_correctable": 0}) for i in range(8)]


class HealthTests(unittest.TestCase):
    def test_historical_ecc_is_not_cleared_or_mistaken_for_new_fault(self):
        m.validate(healthy())
        m.validate(healthy(), baseline=healthy())

    def test_recovery_flags_fail_closed(self):
        for field, value in (("recovery", "Drain and Reset"), ("remap_pending", "Yes"),
                             ("remap_failure", "Yes"), ("ecc_mode", "Disabled")):
            rows = healthy()
            rows[1][field] = value
            with self.assertRaises(ValueError):
                m.validate(rows)

    def test_any_new_counter_or_uncorrectable_fault_rejects(self):
        for field in healthy()[1]["counters"]:
            rows = healthy()
            rows[1]["counters"][field] += 1
            with self.assertRaises(ValueError):
                m.validate(rows, baseline=healthy())
        rows = healthy()
        rows[1]["counters"]["volatile/dram_uncorrectable"] = 1
        with self.assertRaises(ValueError):
            m.validate(rows)

    def test_foreign_process_and_identity_reject(self):
        rows = healthy()
        rows[1]["pids"] = [123]
        with self.assertRaises(ValueError):
            m.validate(rows)
        m.validate(rows, allowed_pids=(123,))
        rows[1]["uuid"] = "different"
        with self.assertRaises(ValueError):
            m.validate(rows, allowed_pids=(123,), baseline=healthy())

    def test_missing_ecc_cannot_pass_as_zero(self):
        rows = healthy()
        rows[1]["counters"] = {}
        with self.assertRaises(ValueError):
            m.validate(rows)

    def test_xml_maps_exact_gpu_and_counts(self):
        row = m.inventory('''<nvidia_smi_log><gpu id="0000:52:00.0">
          <uuid>GPU-one</uuid><product_name>NVIDIA A800 80GB PCIe</product_name>
          <gpu_recovery_action>Drain and Reset</gpu_recovery_action>
          <ecc_mode><current_ecc>Enabled</current_ecc></ecc_mode>
          <ecc_errors><volatile><dram_uncorrectable>32</dram_uncorrectable></volatile>
          <aggregate><dram_uncorrectable>71</dram_uncorrectable>
          <sram_threshold_exceeded>No</sram_threshold_exceeded></aggregate></ecc_errors>
          <remapped_rows><remapped_row_pending>Yes</remapped_row_pending>
          <remapped_row_failure>No</remapped_row_failure></remapped_rows>
          <processes><process_info><pid>456</pid></process_info></processes>
        </gpu></nvidia_smi_log>''')[0]
        self.assertEqual(row["bus"], "0000:52:00.0")
        self.assertEqual(row["counters"]["volatile/dram_uncorrectable"], 32)
        self.assertEqual(row["pids"], [456])


if __name__ == "__main__":
    unittest.main()
