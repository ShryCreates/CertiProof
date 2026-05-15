"""
CertiProof — Field Extractor Unit Tests
==========================================
Run: python test_extractor.py
"""

import unittest
import json
from pathlib import Path

from field_extractor import (
    extract_fields_regex,
    match_institution,
    validate_roll_number,
    merge_extractions,
    FieldExtractor,
)


# ── Sample OCR texts ──────────────────────────────────────────────────────────

GENUINE_TEXT = """
INDIAN INSTITUTE OF TECHNOLOGY BOMBAY
CERTIFICATE OF DEGREE

This is to certify that Arjun Mehta
having satisfactorily completed the requirements
is awarded the degree of Bachelor of Technology
in Computer Science and Engineering

Roll No: 19B050042
CGPA: 8.7 / 10
Date of Issue: 12 June 2023
Certificate No: IITB/2023/BTech/00421
"""

SUSPICIOUS_TEXT = """
DELHI UNIVERSITY
This certifies that Priya Sharma
has been awarded B.A. (Hons) in Economics
Roll Number: 1820041
Grade: 7.4 CGPA
Date: 08 July 2022
"""

FAKE_TEXT = """
ROYAL INDIAN TECH UNIVERSITY
This certifies that Rohit Kumar
M.Tech in AI & ML
Roll No: RIT99999
CGPA: 9.9 / 10
Date: 30 Feb 2024
Certificate No: RITU/FAKE/001
"""


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegexExtraction(unittest.TestCase):

    def test_student_name_extracted(self):
        fields = extract_fields_regex(GENUINE_TEXT)
        self.assertIn("student_name", fields)
        self.assertIn("Arjun", fields["student_name"]["value"])

    def test_degree_extracted(self):
        fields = extract_fields_regex(GENUINE_TEXT)
        self.assertIn("degree", fields)
        val = fields["degree"]["value"].lower()
        self.assertTrue("bachelor" in val or "b.tech" in val or "technology" in val)

    def test_grade_extracted(self):
        fields = extract_fields_regex(GENUINE_TEXT)
        self.assertIn("grade", fields)
        self.assertIn("8.7", fields["grade"]["value"])

    def test_roll_number_extracted(self):
        fields = extract_fields_regex(GENUINE_TEXT)
        self.assertIn("roll_number", fields)
        self.assertEqual(fields["roll_number"]["value"], "19B050042")

    def test_date_extracted(self):
        fields = extract_fields_regex(GENUINE_TEXT)
        self.assertIn("issue_date", fields)
        self.assertIn("2023", fields["issue_date"]["value"])

    def test_confidence_is_float_in_range(self):
        fields = extract_fields_regex(GENUINE_TEXT)
        for field, info in fields.items():
            self.assertIsInstance(info["confidence"], float)
            self.assertGreaterEqual(info["confidence"], 0.0)
            self.assertLessEqual(info["confidence"], 1.0)

    def test_suspicious_text(self):
        fields = extract_fields_regex(SUSPICIOUS_TEXT)
        self.assertIn("student_name", fields)
        self.assertIn("Priya", fields["student_name"]["value"])

    def test_empty_text_returns_empty_dict(self):
        fields = extract_fields_regex("")
        self.assertIsInstance(fields, dict)


class TestInstitutionMatching(unittest.TestCase):

    def test_exact_name_match(self):
        result = match_institution("Indian Institute of Technology Bombay")
        self.assertTrue(result["matched"])
        self.assertEqual(result["code"], "IITB")
        self.assertEqual(result["match_score"], 1.0)

    def test_short_name_match(self):
        result = match_institution("IIT Bombay")
        self.assertTrue(result["matched"])
        self.assertEqual(result["code"], "IITB")

    def test_partial_name_match(self):
        result = match_institution("IIT Madras University")
        self.assertTrue(result["matched"])

    def test_unknown_institution(self):
        result = match_institution("Royal Indian Tech University")
        self.assertFalse(result["matched"])

    def test_empty_string(self):
        result = match_institution("")
        self.assertFalse(result["matched"])

    def test_accreditation_returned(self):
        result = match_institution("IIT Delhi")
        self.assertTrue(result["matched"])
        self.assertIn("NAAC", result["accreditation"])

    def test_roll_pattern_returned(self):
        result = match_institution("Anna University")
        self.assertTrue(result["matched"])
        self.assertIsNotNone(result.get("roll_pattern"))


class TestRollNumberValidation(unittest.TestCase):

    def test_valid_iitb_roll(self):
        inst = match_institution("IIT Bombay")
        self.assertTrue(validate_roll_number("19B050042", inst))

    def test_invalid_roll_wrong_format(self):
        inst = match_institution("IIT Bombay")
        self.assertFalse(validate_roll_number("RIT99999", inst))

    def test_unmatched_institution(self):
        inst = {"matched": False}
        self.assertFalse(validate_roll_number("12345678", inst))

    def test_empty_roll(self):
        inst = match_institution("IIT Bombay")
        self.assertFalse(validate_roll_number("", inst))


class TestMergeExtractions(unittest.TestCase):

    def test_ner_wins_on_higher_confidence(self):
        regex = {"student_name": {"value": "Arjun",      "confidence": 0.70, "method": "regex"}}
        ner   = {"student_name": {"value": "Arjun Mehta","confidence": 0.80, "method": "layoutlmv3"}}
        merged = merge_extractions(regex, ner)
        # NER gets +0.05 boost → 0.85 > 0.70
        self.assertEqual(merged["student_name"]["value"], "Arjun Mehta")

    def test_regex_wins_when_ner_lower(self):
        regex = {"grade": {"value": "8.7 CGPA", "confidence": 0.95, "method": "regex"}}
        ner   = {"grade": {"value": "8.7",       "confidence": 0.60, "method": "layoutlmv3"}}
        merged = merge_extractions(regex, ner)
        # NER boosted to 0.65, still < 0.95
        self.assertEqual(merged["grade"]["value"], "8.7 CGPA")

    def test_ner_only_field_added(self):
        regex = {}
        ner   = {"serial_number": {"value": "CERT001", "confidence": 0.75, "method": "layoutlmv3"}}
        merged = merge_extractions(regex, ner)
        self.assertIn("serial_number", merged)


class TestFieldExtractorWarnings(unittest.TestCase):

    def setUp(self):
        self.extractor = FieldExtractor(use_layoutlmv3=False)

    def test_implausible_cgpa_warning(self):
        fields = {"grade": {"value": "9.9 CGPA", "confidence": 0.8}}
        warnings = self.extractor._validate_fields(fields)
        # 9.9 is valid (≤10), no warning expected
        self.assertEqual(len(warnings), 0)

    def test_cgpa_over_10_warning(self):
        fields = {"grade": {"value": "10.5 CGPA", "confidence": 0.8}}
        warnings = self.extractor._validate_fields(fields)
        self.assertTrue(any("10.5" in w for w in warnings))

    def test_percentage_over_100_warning(self):
        fields = {"grade": {"value": "105%", "confidence": 0.8}}
        warnings = self.extractor._validate_fields(fields)
        self.assertTrue(any("105" in w for w in warnings))

    def test_invalid_year_warning(self):
        fields = {"issue_date": {"value": "30 Feb 2099", "confidence": 0.8}}
        warnings = self.extractor._validate_fields(fields)
        self.assertTrue(any("2099" in w for w in warnings))

    def test_single_word_name_warning(self):
        fields = {"student_name": {"value": "Arjun", "confidence": 0.8}}
        warnings = self.extractor._validate_fields(fields)
        self.assertTrue(any("incomplete" in w.lower() for w in warnings))


class TestSampleOutputShape(unittest.TestCase):
    """Verify sample_output.json matches expected schema."""

    def test_sample_output_schema(self):
        sample_path = Path(__file__).parent / "sample_output.json"
        self.assertTrue(sample_path.exists(), "sample_output.json not found")

        with open(sample_path) as f:
            data = json.load(f)

        self.assertIn("fields", data)
        self.assertIn("institution_match", data)
        self.assertIn("roll_valid", data)
        self.assertIn("overall_confidence", data)
        self.assertIn("warnings", data)

        for field_name, field_data in data["fields"].items():
            self.assertIn("value",      field_data, f"{field_name} missing 'value'")
            self.assertIn("confidence", field_data, f"{field_name} missing 'confidence'")

        im = data["institution_match"]
        self.assertIn("matched", im)
        if im["matched"]:
            self.assertIn("name",          im)
            self.assertIn("code",          im)
            self.assertIn("accreditation", im)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CertiProof — Field Extractor Tests")
    print("=" * 60)
    unittest.main(verbosity=2)
