"""Bounded arithmetic AST interpretation, never eval/exec."""

import ast
import re
from decimal import Decimal, DecimalException, localcontext

from agentic_rag.tools.models import CalculateInput, CalculateOutput


class CalculationError(ValueError):
    pass


def calculate(arguments: CalculateInput) -> CalculateOutput:
    expression = arguments.expression.strip()
    if not re.fullmatch(r"[0-9.\s+*/()-]+", expression):
        raise CalculationError("Only decimal arithmetic is supported")
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise CalculationError("Expression is too complex")

        def visit(node: ast.AST) -> Decimal:
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                literal = ast.get_source_segment(expression, node)
                if literal is None:
                    raise CalculationError("Invalid number")
                value = Decimal(literal)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                operand = visit(node.operand)
                value = operand if isinstance(node.op, ast.UAdd) else -operand
            elif isinstance(node, ast.BinOp):
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add):
                    value = left + right
                elif isinstance(node.op, ast.Sub):
                    value = left - right
                elif isinstance(node.op, ast.Mult):
                    value = left * right
                elif isinstance(node.op, ast.Div):
                    value = left / right
                else:
                    raise CalculationError("Unsupported operator")
            else:
                raise CalculationError("Unsupported expression")
            if not value.is_finite() or (value and not -100 <= value.adjusted() <= 100):
                raise CalculationError("Result magnitude exceeds limit")
            return value

        with localcontext() as context:
            context.prec = 28
            value = +visit(tree.body)
        return CalculateOutput(expression=expression, value=str(value))
    except (SyntaxError, DecimalException, RecursionError) as exc:
        raise CalculationError("Invalid arithmetic or division by zero") from exc
