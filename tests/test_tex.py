import pytest

from ctfeval.tex import number2tex, multirow


@pytest.mark.parametrize(
    "name,value,digits,expected",
    [
        ("var1", 3.1415, 0, "3"),
        ("var2", 3.1415, 2, "3.14"),
        ("var3", 3.1415, 4, "3.1415"),
    ],
)
def test_number2tex(name, value, digits, expected):
    result = number2tex(name, value, digits)
    assert "\\newcommand" in result
    assert f"{{\\{name}}}" in result
    assert f"{{{expected}}}" in result


def test_multirow():
    values = ["raw", "raw", "raw", "raw", "delta"]
    skip1 = {"raw": 2}
    skip2 = {"raw": 4}

    result1 = multirow(values, skip1)
    assert result1 == [
        "\\multirow{2}{*}{raw}",
        "",
        "\\multirow{2}{*}{raw}",
        "",
        "delta",
    ]

    result2 = multirow(values, skip2)
    assert result2 == ["\\multirow{4}{*}{raw}", "", "", "", "delta"]
