import pytest
from pydantic import ValidationError

from agentic_rag.tools.calculator import CalculationError, calculate
from agentic_rag.tools.models import CalculateInput, CatalogInput, SearchInput, tool_definitions


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("6 / 8 * 100", "75.00"),
        ("0.1+0.2", "0.3"),
        ("-(2+3)*4", "-20"),
        ("1/3", "0.3333333333333333333333333333"),
    ],
)
def test_decimal_arithmetic(expression: str, expected: str) -> None:
    assert calculate(CalculateInput(expression=expression)).value == expected


@pytest.mark.parametrize(
    "expression",
    [
        "1/0",
        "2**99999",
        "2//1",
        "__import__('os')",
        "a+1",
        "[1]",
        "True",
        "1e9999",
        "9" * 110,
        "1+" * 40 + "1",
    ],
)
def test_calculator_rejects_code_and_excessive_work(expression: str) -> None:
    with pytest.raises(CalculationError):
        calculate(CalculateInput(expression=expression))


def test_tool_contracts_are_strict_and_no_arbitrary_sql() -> None:
    with pytest.raises(ValidationError):
        SearchInput.model_validate_json('{"query":"x","k":"2"}')
    with pytest.raises(ValidationError):
        CatalogInput.model_validate_json('{"sql":"DROP TABLE documents"}')
    with pytest.raises(ValidationError):
        SearchInput(query="  ")
    assert len(tool_definitions()) == 3
