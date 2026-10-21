#!/usr/bin/env python3
"""
DocVerify - Russian Internal Passport Field Extraction Regression Test
Author: Senior ML & OCR Systems Engineer

Validates:
1. Cyrillic text extraction without transliteration or character corruption.
2. Correct field alignment for surname, name, patronymic, sex, birth date, birth place, and number.
3. Date format validation (DD.MM.YYYY).
4. Honest confidence score reporting.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from src.inference_engine import DocVerifyInferenceEngine

def test_russian_passport_extraction():
    sys.stdout.reconfigure(encoding='utf-8')
    print("Testing Russian Internal Passport Field Extraction...")
    
    engine = DocVerifyInferenceEngine(enable_ocr=True)
    sample_path = PROJECT_ROOT / "Dataset" / "template" / "Images" / "reals" / "rus_internalpassport_00.jpg"
    
    if not sample_path.exists():
        print(f"Sample not found: {sample_path}")
        return False
        
    img = Image.open(sample_path)
    res = engine.verify_document(img, template_override="rus_internalpassport")
    
    fields = {f["field_name"]: f for f in res["ocr_findings"]}
    
    expected_fields = ["surname", "name", "patronymic", "sex", "birth_date", "birth_place", "number"]
    for ef in expected_fields:
        assert ef in fields, f"Missing expected field: {ef}"
        
    # Check Cyrillic values and date
    assert fields["surname"]["text"] == "КАРИБЖАНОВ", f"Unexpected surname: {fields['surname']['text']}"
    assert fields["name"]["text"] == "ЛЕОНИД", f"Unexpected name: {fields['name']['text']}"
    assert fields["patronymic"]["text"] == "ДАВИДОВИЧ", f"Unexpected patronymic: {fields['patronymic']['text']}"
    assert fields["sex"]["text"] == "МУЖ", f"Unexpected sex: {fields['sex']['text']}"
    assert fields["birth_date"]["text"] == "03.06.1978", f"Unexpected birth_date: {fields['birth_date']['text']}"
    assert "МОСКВА" in fields["birth_place"]["text"], f"Unexpected birth_place: {fields['birth_place']['text']}"
    assert "45 98 647461" in fields["number"]["text"], f"Unexpected number: {fields['number']['text']}"
    
    # Check honest confidence scores
    for name, f in fields.items():
        assert f["confidence"] > 0.5, f"Confidence too low for {name}: {f['confidence']}"
        assert not f["is_anomaly"], f"False anomaly detected for valid field {name}"
        
    print("ALL RUSSIAN INTERNAL PASSPORT EXTRACTION CHECKS PASSED!")
    return True

if __name__ == "__main__":
    success = test_russian_passport_extraction()
    if not success:
        sys.exit(1)
