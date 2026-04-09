#!/usr/bin/env python3
"""Reusable node-based fractal genome primitives for recursive strategy search."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class ConditionSpec:
    feature: str
    comparator: Literal[">=", "<="]
    threshold: float
    invert: bool = False


@dataclass(frozen=True)
class ThresholdCell:
    spec: ConditionSpec


@dataclass(frozen=True)
class LeafGene:
    route_threshold_bias: int = 0
    mapping_shift: int = 0
    target_vol_scale: float = 1.0
    gross_cap_scale: float = 1.0
    kill_switch_scale: float = 1.0
    cooldown_scale: float = 1.0


@dataclass
class AndCell:
    left: "LogicCell"
    right: "LogicCell"


@dataclass
class OrCell:
    left: "LogicCell"
    right: "LogicCell"


@dataclass
class NotCell:
    child: "LogicCell"


LogicCell = ThresholdCell | AndCell | OrCell | NotCell


@dataclass(frozen=True)
class LeafNode:
    expert_idx: int
    gene: LeafGene = field(default_factory=LeafGene)


@dataclass
class ConditionNode:
    condition: LogicCell
    if_true: "TreeNode"
    if_false: "TreeNode"


TreeNode = LeafNode | ConditionNode


@dataclass(frozen=True)
class FilterDecision:
    accepted: bool
    source: str
    reason: str
    llm_prompt: str | None = None


def serialize_logic(cell: LogicCell) -> dict[str, Any]:
    if isinstance(cell, ThresholdCell):
        return {
            "type": "threshold",
            "feature": cell.spec.feature,
            "comparator": cell.spec.comparator,
            "threshold": float(cell.spec.threshold),
            "invert": bool(cell.spec.invert),
        }
    if isinstance(cell, AndCell):
        return {"type": "and", "left": serialize_logic(cell.left), "right": serialize_logic(cell.right)}
    if isinstance(cell, OrCell):
        return {"type": "or", "left": serialize_logic(cell.left), "right": serialize_logic(cell.right)}
    return {"type": "not", "child": serialize_logic(cell.child)}


def serialize_leaf_gene(gene: LeafGene) -> dict[str, Any]:
    return {
        "route_threshold_bias": int(gene.route_threshold_bias),
        "mapping_shift": int(gene.mapping_shift),
        "target_vol_scale": float(gene.target_vol_scale),
        "gross_cap_scale": float(gene.gross_cap_scale),
        "kill_switch_scale": float(gene.kill_switch_scale),
        "cooldown_scale": float(gene.cooldown_scale),
    }


def deserialize_leaf_gene(raw: dict[str, Any] | None) -> LeafGene:
    if not raw:
        return LeafGene()
    return LeafGene(
        route_threshold_bias=int(raw.get("route_threshold_bias", 0)),
        mapping_shift=int(raw.get("mapping_shift", 0)),
        target_vol_scale=float(raw.get("target_vol_scale", 1.0)),
        gross_cap_scale=float(raw.get("gross_cap_scale", 1.0)),
        kill_switch_scale=float(raw.get("kill_switch_scale", 1.0)),
        cooldown_scale=float(raw.get("cooldown_scale", 1.0)),
    )


def deserialize_logic(raw: dict[str, Any]) -> LogicCell:
    kind = str(raw["type"])
    if kind == "threshold":
        return ThresholdCell(
            spec=ConditionSpec(
                feature=str(raw["feature"]),
                comparator=str(raw["comparator"]),
                threshold=float(raw["threshold"]),
                invert=bool(raw.get("invert", False)),
            )
        )
    if kind == "and":
        return AndCell(left=deserialize_logic(raw["left"]), right=deserialize_logic(raw["right"]))
    if kind == "or":
        return OrCell(left=deserialize_logic(raw["left"]), right=deserialize_logic(raw["right"]))
    if kind == "not":
        return NotCell(child=deserialize_logic(raw["child"]))
    raise ValueError(f"Unknown logic cell type: {kind}")


def serialize_tree(node: TreeNode) -> dict[str, Any]:
    if isinstance(node, LeafNode):
        payload = {"type": "leaf", "expert_idx": int(node.expert_idx)}
        if node.gene != LeafGene():
            payload["gene"] = serialize_leaf_gene(node.gene)
        return payload
    return {
        "type": "condition",
        "condition": serialize_logic(node.condition),
        "if_true": serialize_tree(node.if_true),
        "if_false": serialize_tree(node.if_false),
    }


def deserialize_tree(raw: dict[str, Any]) -> TreeNode:
    if raw["type"] == "leaf":
        return LeafNode(int(raw["expert_idx"]), deserialize_leaf_gene(raw.get("gene")))
    if "condition" in raw:
        condition = deserialize_logic(raw["condition"])
    else:
        condition = ThresholdCell(
            spec=ConditionSpec(
                feature=str(raw["feature"]),
                comparator=str(raw["comparator"]),
                threshold=float(raw["threshold"]),
                invert=bool(raw.get("invert", False)),
            )
        )
    return ConditionNode(
        condition=condition,
        if_true=deserialize_tree(raw["if_true"]),
        if_false=deserialize_tree(raw["if_false"]),
    )


def tree_key(node: TreeNode) -> str:
    return json.dumps(serialize_tree(node), sort_keys=True, ensure_ascii=False)


def logic_key(cell: LogicCell) -> str:
    return json.dumps(serialize_logic(cell), sort_keys=True, ensure_ascii=False)


def leaf_key(node: LeafNode) -> str:
    return json.dumps(
        {
            "expert_idx": int(node.expert_idx),
            "gene": serialize_leaf_gene(node.gene),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def logic_depth(cell: LogicCell) -> int:
    if isinstance(cell, ThresholdCell):
        return 0
    if isinstance(cell, NotCell):
        return 1 + logic_depth(cell.child)
    return 1 + max(logic_depth(cell.left), logic_depth(cell.right))


def logic_size(cell: LogicCell) -> int:
    if isinstance(cell, ThresholdCell):
        return 1
    if isinstance(cell, NotCell):
        return 1 + logic_size(cell.child)
    return 1 + logic_size(cell.left) + logic_size(cell.right)


def tree_depth(node: TreeNode) -> int:
    if isinstance(node, LeafNode):
        return 0
    return 1 + max(tree_depth(node.if_true), tree_depth(node.if_false))


def tree_size(node: TreeNode) -> int:
    if isinstance(node, LeafNode):
        return 1
    return 1 + tree_size(node.if_true) + tree_size(node.if_false)


def tree_logic_size(node: TreeNode) -> int:
    if isinstance(node, LeafNode):
        return 0
    return logic_size(node.condition) + tree_logic_size(node.if_true) + tree_logic_size(node.if_false)


def tree_logic_depth(node: TreeNode) -> int:
    if isinstance(node, LeafNode):
        return 0
    return max(logic_depth(node.condition), tree_logic_depth(node.if_true), tree_logic_depth(node.if_false))


def collect_leaves(node: TreeNode, out: list[int] | None = None) -> list[int]:
    if out is None:
        out = []
    if isinstance(node, LeafNode):
        out.append(int(node.expert_idx))
        return out
    collect_leaves(node.if_true, out)
    collect_leaves(node.if_false, out)
    return out


def collect_leaf_keys(node: TreeNode, out: list[str] | None = None) -> list[str]:
    if out is None:
        out = []
    if isinstance(node, LeafNode):
        out.append(leaf_key(node))
        return out
    collect_leaf_keys(node.if_true, out)
    collect_leaf_keys(node.if_false, out)
    return out


def collect_leaf_nodes(node: TreeNode, out: list[LeafNode] | None = None) -> list[LeafNode]:
    if out is None:
        out = []
    if isinstance(node, LeafNode):
        out.append(node)
        return out
    collect_leaf_nodes(node.if_true, out)
    collect_leaf_nodes(node.if_false, out)
    return out


def collect_logic_specs(cell: LogicCell, out: list[ConditionSpec] | None = None) -> list[ConditionSpec]:
    if out is None:
        out = []
    if isinstance(cell, ThresholdCell):
        out.append(cell.spec)
        return out
    if isinstance(cell, NotCell):
        collect_logic_specs(cell.child, out)
        return out
    collect_logic_specs(cell.left, out)
    collect_logic_specs(cell.right, out)
    return out


def collect_specs(node: TreeNode, out: list[ConditionSpec] | None = None) -> list[ConditionSpec]:
    if out is None:
        out = []
    if isinstance(node, LeafNode):
        return out
    collect_logic_specs(node.condition, out)
    collect_specs(node.if_true, out)
    collect_specs(node.if_false, out)
    return out


def condition_to_text(spec: ConditionSpec) -> str:
    clause = f"{spec.feature} {spec.comparator} {spec.threshold:.4f}"
    return f"NOT({clause})" if spec.invert else clause


def logic_to_text(cell: LogicCell) -> str:
    if isinstance(cell, ThresholdCell):
        return condition_to_text(cell.spec)
    if isinstance(cell, NotCell):
        return f"NOT({logic_to_text(cell.child)})"
    op = "AND" if isinstance(cell, AndCell) else "OR"
    return f"({logic_to_text(cell.left)} {op} {logic_to_text(cell.right)})"


def tree_to_text(node: TreeNode, depth: int = 0) -> str:
    pad = "  " * depth
    if isinstance(node, LeafNode):
        if node.gene == LeafGene():
            return f"{pad}LEAF expert[{int(node.expert_idx)}]"
        return (
            f"{pad}LEAF expert[{int(node.expert_idx)}] "
            f"gene={json.dumps(serialize_leaf_gene(node.gene), ensure_ascii=False, sort_keys=True)}"
        )
    lines = [f"{pad}IF {logic_to_text(node.condition)}"]
    lines.append(tree_to_text(node.if_true, depth + 1))
    lines.append(f"{pad}else")
    lines.append(tree_to_text(node.if_false, depth + 1))
    return "\n".join(lines)


def build_llm_prompt(node: TreeNode, expert_pool: list[dict[str, Any]]) -> str:
    lines = [
        "다음 트레이딩 트리의 경제적 타당성을 점검하라.",
        "목표: 자기모순, 의미 없는 중복 조건, 동일 branch, leaf 편중, 해석 불가능한 조합을 걸러내라.",
        "트리(TEXT):",
        tree_to_text(node),
        "트리(JSON):",
        json.dumps(serialize_tree(node), ensure_ascii=False),
        "leaf 전문가 요약:",
    ]
    for idx, expert in enumerate(expert_pool):
        score = float(expert.get("score", 0.0))
        pair_configs = expert.get("pair_configs", {})
        lines.append(f"- expert {idx}: score={score:.4f} pair_configs={json.dumps(pair_configs, ensure_ascii=False)}")
    return "\n".join(lines)


def load_llm_review_map(path: str | None) -> dict[str, FilterDecision]:
    if not path:
        return {}
    review_path = Path(path)
    if not review_path.exists():
        return {}
    out: dict[str, FilterDecision] = {}
    for line in review_path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        out[str(raw["tree_key"])] = FilterDecision(
            accepted=bool(raw["accepted"]),
            source=str(raw.get("source", "llm_review")),
            reason=str(raw.get("reason", "reviewed")),
            llm_prompt=raw.get("llm_prompt"),
        )
    return out


def _spec_signature(spec: ConditionSpec) -> tuple[str, str, float]:
    return (spec.feature, spec.comparator, float(spec.threshold))


def _cell_has_degenerate_logic(cell: LogicCell) -> bool:
    if isinstance(cell, ThresholdCell):
        return False
    if isinstance(cell, NotCell):
        if isinstance(cell.child, NotCell):
            return True
        return _cell_has_degenerate_logic(cell.child)
    left_serialized = logic_key(cell.left)
    right_serialized = logic_key(cell.right)
    if left_serialized == right_serialized:
        return True
    if (
        isinstance(cell.left, ThresholdCell)
        and isinstance(cell.right, ThresholdCell)
        and _spec_signature(cell.left.spec) == _spec_signature(cell.right.spec)
        and cell.left.spec.invert != cell.right.spec.invert
    ):
        return True
    return _cell_has_degenerate_logic(cell.left) or _cell_has_degenerate_logic(cell.right)


def _tree_has_degenerate_logic(node: TreeNode) -> bool:
    if isinstance(node, LeafNode):
        return False
    if _cell_has_degenerate_logic(node.condition):
        return True
    return _tree_has_degenerate_logic(node.if_true) or _tree_has_degenerate_logic(node.if_false)


def heuristic_semantic_filter(node: TreeNode, expert_pool: list[dict[str, Any]], max_depth: int) -> FilterDecision:
    leaf_ids = collect_leaf_keys(node)
    specs = collect_specs(node)
    prompt = build_llm_prompt(node, expert_pool)

    if tree_depth(node) > max_depth:
        return FilterDecision(False, "heuristic", "tree_depth_exceeded", prompt)
    if not specs:
        return FilterDecision(False, "heuristic", "no_conditions", prompt)
    if len(set(leaf_ids)) < 2:
        return FilterDecision(False, "heuristic", "single_expert_only", prompt)
    if tree_size(node) > 15:
        return FilterDecision(False, "heuristic", "tree_too_large", prompt)
    if tree_logic_size(node) > 27:
        return FilterDecision(False, "heuristic", "logic_cell_budget_exceeded", prompt)

    unique_specs = {(s.feature, s.comparator, s.threshold, s.invert) for s in specs}
    if len(specs) >= 2 and len(unique_specs) <= 1:
        return FilterDecision(False, "heuristic", "duplicated_conditions_only", prompt)

    def has_identical_branches(current: TreeNode) -> bool:
        if isinstance(current, LeafNode):
            return False
        if serialize_tree(current.if_true) == serialize_tree(current.if_false):
            return True
        return has_identical_branches(current.if_true) or has_identical_branches(current.if_false)

    if has_identical_branches(node):
        return FilterDecision(False, "heuristic", "identical_branches", prompt)
    if _tree_has_degenerate_logic(node):
        return FilterDecision(False, "heuristic", "degenerate_logic_cell", prompt)

    if len(leaf_ids) >= 4:
        dominant = max(leaf_ids.count(value) for value in set(leaf_ids)) / len(leaf_ids)
        if dominant > 0.75:
            return FilterDecision(False, "heuristic", "leaf_concentration_too_high", prompt)

    return FilterDecision(True, "heuristic_fallback", "accepted", prompt)


def semantic_filter(
    node: TreeNode,
    expert_pool: list[dict[str, Any]],
    max_depth: int,
    mode: str,
    llm_reviews: dict[str, FilterDecision] | None = None,
) -> FilterDecision:
    reviews = llm_reviews or {}
    key = tree_key(node)
    reviewed = reviews.get(key)
    if mode == "llm-only":
        if reviewed is not None:
            return reviewed
        return FilterDecision(False, "llm_only", "missing_llm_review", build_llm_prompt(node, expert_pool))
    heuristic = heuristic_semantic_filter(node, expert_pool, max_depth)
    if mode in {"llm-first", "auto"} and reviewed is not None:
        return reviewed
    return heuristic


LEAF_ROUTE_BIAS_CHOICES = (-1, 0, 1)
LEAF_MAPPING_SHIFT_CHOICES = (-8, -4, 0, 4, 8)
LEAF_SCALE_CHOICES = (0.85, 1.0, 1.15)
LEAF_COOLDOWN_SCALE_CHOICES = (0.75, 1.0, 1.25)


def random_leaf_gene(rng: random.Random) -> LeafGene:
    if rng.random() < 0.45:
        return LeafGene()
    return LeafGene(
        route_threshold_bias=rng.choice(LEAF_ROUTE_BIAS_CHOICES),
        mapping_shift=rng.choice(LEAF_MAPPING_SHIFT_CHOICES),
        target_vol_scale=rng.choice(LEAF_SCALE_CHOICES),
        gross_cap_scale=rng.choice(LEAF_SCALE_CHOICES),
        kill_switch_scale=rng.choice(LEAF_SCALE_CHOICES),
        cooldown_scale=rng.choice(LEAF_COOLDOWN_SCALE_CHOICES),
    )


def mutate_leaf_gene(gene: LeafGene, rng: random.Random) -> LeafGene:
    field_name = rng.choice(
        (
            "route_threshold_bias",
            "mapping_shift",
            "target_vol_scale",
            "gross_cap_scale",
            "kill_switch_scale",
            "cooldown_scale",
        )
    )
    if field_name == "route_threshold_bias":
        return LeafGene(
            route_threshold_bias=rng.choice(LEAF_ROUTE_BIAS_CHOICES),
            mapping_shift=gene.mapping_shift,
            target_vol_scale=gene.target_vol_scale,
            gross_cap_scale=gene.gross_cap_scale,
            kill_switch_scale=gene.kill_switch_scale,
            cooldown_scale=gene.cooldown_scale,
        )
    if field_name == "mapping_shift":
        return LeafGene(
            route_threshold_bias=gene.route_threshold_bias,
            mapping_shift=rng.choice(LEAF_MAPPING_SHIFT_CHOICES),
            target_vol_scale=gene.target_vol_scale,
            gross_cap_scale=gene.gross_cap_scale,
            kill_switch_scale=gene.kill_switch_scale,
            cooldown_scale=gene.cooldown_scale,
        )
    if field_name == "target_vol_scale":
        return LeafGene(
            route_threshold_bias=gene.route_threshold_bias,
            mapping_shift=gene.mapping_shift,
            target_vol_scale=rng.choice(LEAF_SCALE_CHOICES),
            gross_cap_scale=gene.gross_cap_scale,
            kill_switch_scale=gene.kill_switch_scale,
            cooldown_scale=gene.cooldown_scale,
        )
    if field_name == "gross_cap_scale":
        return LeafGene(
            route_threshold_bias=gene.route_threshold_bias,
            mapping_shift=gene.mapping_shift,
            target_vol_scale=gene.target_vol_scale,
            gross_cap_scale=rng.choice(LEAF_SCALE_CHOICES),
            kill_switch_scale=gene.kill_switch_scale,
            cooldown_scale=gene.cooldown_scale,
        )
    if field_name == "kill_switch_scale":
        return LeafGene(
            route_threshold_bias=gene.route_threshold_bias,
            mapping_shift=gene.mapping_shift,
            target_vol_scale=gene.target_vol_scale,
            gross_cap_scale=gene.gross_cap_scale,
            kill_switch_scale=rng.choice(LEAF_SCALE_CHOICES),
            cooldown_scale=gene.cooldown_scale,
        )
    return LeafGene(
        route_threshold_bias=gene.route_threshold_bias,
        mapping_shift=gene.mapping_shift,
        target_vol_scale=gene.target_vol_scale,
        gross_cap_scale=gene.gross_cap_scale,
        kill_switch_scale=gene.kill_switch_scale,
        cooldown_scale=rng.choice(LEAF_COOLDOWN_SCALE_CHOICES),
    )


def random_leaf(rng: random.Random, expert_count: int) -> LeafNode:
    return LeafNode(expert_idx=rng.randrange(expert_count), gene=random_leaf_gene(rng))


def random_logic_cell(
    rng: random.Random,
    condition_options: list[ConditionSpec],
    max_depth: int,
    depth: int = 0,
) -> LogicCell:
    if depth >= max_depth or (depth > 0 and rng.random() < 0.50):
        return ThresholdCell(spec=copy.deepcopy(rng.choice(condition_options)))
    choice = rng.choices(("and", "or", "not"), weights=(0.45, 0.35, 0.20), k=1)[0]
    if choice == "not":
        return NotCell(child=random_logic_cell(rng, condition_options, max_depth, depth + 1))
    left = random_logic_cell(rng, condition_options, max_depth, depth + 1)
    right = random_logic_cell(rng, condition_options, max_depth, depth + 1)
    if choice == "and":
        return AndCell(left=left, right=right)
    return OrCell(left=left, right=right)


def random_tree(
    rng: random.Random,
    condition_options: list[ConditionSpec],
    expert_count: int,
    max_depth: int,
    depth: int = 0,
    logic_max_depth: int | None = None,
) -> TreeNode:
    resolved_logic_depth = max(1, logic_max_depth if logic_max_depth is not None else min(3, max_depth))
    if depth >= max_depth or (depth > 0 and rng.random() < 0.35):
        return random_leaf(rng, expert_count)
    return ConditionNode(
        condition=random_logic_cell(rng, condition_options, resolved_logic_depth),
        if_true=random_tree(rng, condition_options, expert_count, max_depth, depth + 1, resolved_logic_depth),
        if_false=random_tree(rng, condition_options, expert_count, max_depth, depth + 1, resolved_logic_depth),
    )


def subtree_paths(node: TreeNode, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], TreeNode]]:
    out = [(path, node)]
    if isinstance(node, ConditionNode):
        out.extend(subtree_paths(node.if_true, path + ("if_true",)))
        out.extend(subtree_paths(node.if_false, path + ("if_false",)))
    return out


def logic_paths(cell: LogicCell, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], LogicCell]]:
    out = [(path, cell)]
    if isinstance(cell, NotCell):
        out.extend(logic_paths(cell.child, path + ("child",)))
    elif isinstance(cell, (AndCell, OrCell)):
        out.extend(logic_paths(cell.left, path + ("left",)))
        out.extend(logic_paths(cell.right, path + ("right",)))
    return out


def set_subtree(node: TreeNode, path: tuple[str, ...], new_subtree: TreeNode) -> TreeNode:
    if not path:
        return copy.deepcopy(new_subtree)
    root = copy.deepcopy(node)
    current = root
    for step in path[:-1]:
        current = getattr(current, step)
    setattr(current, path[-1], copy.deepcopy(new_subtree))
    return root


def set_logic_subtree(cell: LogicCell, path: tuple[str, ...], new_subtree: LogicCell) -> LogicCell:
    if not path:
        return copy.deepcopy(new_subtree)
    root = copy.deepcopy(cell)
    current = root
    for step in path[:-1]:
        current = getattr(current, step)
    setattr(current, path[-1], copy.deepcopy(new_subtree))
    return root


def mutate_logic_cell(
    cell: LogicCell,
    rng: random.Random,
    condition_options: list[ConditionSpec],
    max_depth: int,
) -> LogicCell:
    target_path, target_cell = rng.choice(logic_paths(cell))
    mode = rng.choice(("replace_subtree", "tweak_threshold", "flip_operator"))
    if mode == "replace_subtree":
        return set_logic_subtree(cell, target_path, random_logic_cell(rng, condition_options, max_depth))
    if isinstance(target_cell, ThresholdCell) and mode == "tweak_threshold":
        return set_logic_subtree(
            cell,
            target_path,
            ThresholdCell(spec=copy.deepcopy(rng.choice(condition_options))),
        )
    if isinstance(target_cell, AndCell) and mode == "flip_operator":
        return set_logic_subtree(cell, target_path, OrCell(left=target_cell.left, right=target_cell.right))
    if isinstance(target_cell, OrCell) and mode == "flip_operator":
        return set_logic_subtree(cell, target_path, AndCell(left=target_cell.left, right=target_cell.right))
    if isinstance(target_cell, NotCell) and mode == "flip_operator":
        return set_logic_subtree(cell, target_path, copy.deepcopy(target_cell.child))
    return set_logic_subtree(cell, target_path, random_logic_cell(rng, condition_options, max_depth))


def mutate_tree(
    node: TreeNode,
    rng: random.Random,
    condition_options: list[ConditionSpec],
    expert_count: int,
    max_depth: int,
    logic_max_depth: int | None = None,
) -> TreeNode:
    resolved_logic_depth = max(1, logic_max_depth if logic_max_depth is not None else min(3, max_depth))
    target_path, target_node = rng.choice(subtree_paths(node))
    mode = rng.choice(("replace_subtree", "tweak_condition", "tweak_leaf"))
    mutated = copy.deepcopy(node)
    if mode == "replace_subtree":
        return set_subtree(
            mutated,
            target_path,
            random_tree(rng, condition_options, expert_count, max_depth, logic_max_depth=resolved_logic_depth),
        )
    if isinstance(target_node, ConditionNode) and mode == "tweak_condition":
        replacement = copy.deepcopy(target_node)
        replacement.condition = mutate_logic_cell(target_node.condition, rng, condition_options, resolved_logic_depth)
        return set_subtree(mutated, target_path, replacement)
    if isinstance(target_node, LeafNode):
        if rng.random() < 0.50:
            replacement = LeafNode(rng.randrange(expert_count), target_node.gene)
        else:
            replacement = LeafNode(target_node.expert_idx, mutate_leaf_gene(target_node.gene, rng))
        return set_subtree(mutated, target_path, replacement)
    return set_subtree(mutated, target_path, random_leaf(rng, expert_count))


def crossover_tree(parent_a: TreeNode, parent_b: TreeNode, rng: random.Random) -> tuple[TreeNode, TreeNode]:
    path_a, subtree_a = rng.choice(subtree_paths(parent_a))
    path_b, subtree_b = rng.choice(subtree_paths(parent_b))
    return set_subtree(parent_a, path_a, subtree_b), set_subtree(parent_b, path_b, subtree_a)


def feature_condition_values(spec: ConditionSpec, features: dict[str, np.ndarray]) -> np.ndarray:
    values = features[spec.feature]
    if spec.comparator == ">=":
        result = values >= spec.threshold
    else:
        result = values <= spec.threshold
    if spec.invert:
        result = ~result
    return result


def evaluate_logic_cell(cell: LogicCell, features: dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(cell, ThresholdCell):
        return feature_condition_values(cell.spec, features)
    if isinstance(cell, NotCell):
        return ~evaluate_logic_cell(cell.child, features)
    if isinstance(cell, AndCell):
        return evaluate_logic_cell(cell.left, features) & evaluate_logic_cell(cell.right, features)
    return evaluate_logic_cell(cell.left, features) | evaluate_logic_cell(cell.right, features)


def evaluate_tree_codes(node: TreeNode, features: dict[str, np.ndarray]) -> np.ndarray:
    first = next(iter(features.values()))
    out = np.zeros(len(first), dtype="int16")

    def fill(current: TreeNode, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        if isinstance(current, LeafNode):
            out[mask] = int(current.expert_idx)
            return
        cond = evaluate_logic_cell(current.condition, features)
        fill(current.if_true, mask & cond)
        fill(current.if_false, mask & (~cond))

    fill(node, np.ones(len(first), dtype=bool))
    return out


def compile_leaf_catalog(node: TreeNode) -> tuple[list[LeafNode], dict[str, int]]:
    catalog: list[LeafNode] = []
    index_by_key: dict[str, int] = {}
    for leaf in collect_leaf_nodes(node):
        key = leaf_key(leaf)
        if key in index_by_key:
            continue
        index_by_key[key] = len(catalog)
        catalog.append(copy.deepcopy(leaf))
    return catalog, index_by_key


def evaluate_tree_leaf_codes(node: TreeNode, features: dict[str, np.ndarray]) -> tuple[np.ndarray, list[LeafNode]]:
    first = next(iter(features.values()))
    out = np.zeros(len(first), dtype="int16")
    catalog, index_by_key = compile_leaf_catalog(node)

    def fill(current: TreeNode, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        if isinstance(current, LeafNode):
            out[mask] = int(index_by_key[leaf_key(current)])
            return
        cond = evaluate_logic_cell(current.condition, features)
        fill(current.if_true, mask & cond)
        fill(current.if_false, mask & (~cond))

    fill(node, np.ones(len(first), dtype=bool))
    return out, catalog
