import sys; sys.path.insert(0, "eval")
from decimal import Decimal
from compare import results_match

def test_decimal_vs_float():   assert results_match([(Decimal("0.2858"),)], [(0.28580001,)])
def test_case_and_space():     assert results_match([("Gràcia",)], [(" gràcia ",)])
def test_row_order_ignored():  assert results_match([(1,), (2,)], [(2,), (1,)])
def test_row_order_enforced(): assert not results_match([(1,), (2,)], [(2,), (1,)], ordered=True)
def test_column_order():       assert results_match([("a", 1)], [(1, "a")])
def test_extra_column_fails(): assert not results_match([("a",)], [("a", 1)])
def test_wrong_value():        assert not results_match([(5,)], [(6,)])
