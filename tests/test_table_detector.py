import pytest
import os

from src.table_detector import TableDetector

class TestTableDetection:
    @pytest.fixture(scope="class")
    def table_detector(self):
        return TableDetector()


    def test_successful_invoice_extraction(self,table_detector):
        result = table_detector.predict("tests/images/invoice.png")
        assert len(result) > 0
        assert any(detection["label"] == "table" for detection in result)

        img = table_detector.draw_boxes(result,"tests/images/invoice.png")
        img.show()

    def test_successful_bank_document_extraction(self,table_detector):
        result = table_detector.predict("tests/images/bank_document.jpg")
        assert len(result) > 0
        assert any(detection["label"] == "table" for detection in result)

        img = table_detector.draw_boxes(result,"tests/images/bank_document.jpg")
        img.show()

    def test_empty_document(self,table_detector):
        result = table_detector.predict("tests/images/empty_document.jpg")
        assert len(result) == 0

    def test_multiple_tables_detection(self,table_detector):
        result = table_detector.predict("tests/images/multiple_tables.png")
        assert len([detection for detection in result if detection["label"] == "table"]) > 1

        img = table_detector.draw_boxes(result,"tests/images/multiple_tables.png")
        img.show()
    
    def test_error_handling_nonexistent_file(self,table_detector):
        with pytest.raises(ValueError):
            table_detector.predict("tests/images/nonexistent_file.jpg")

    def test_error_handling_invalid_image(self,table_detector):
        with pytest.raises(ValueError):
            table_detector.predict("tests/images/invalid_image.txt")



