#!/usr/bin/env python3
"""Reusable node-based fractal genome primitives for recursive strategy search."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class ConditionSpec:
    feature: str
    comparator: Literal[">=", "<="]
    threshold: float
    invert: bool = False


@dataclass
class LeafNode:
    expert_idx: int


@dataclass
class ConditionNode:
    spec: ConditionSpec
    if_true: "TreeNode"
    if_false: "TreeNode"


TreeNode = LeafNode | ConditionNode


@dataclass(frozen=True)
class FilterDecision:
    accepted: bool
    source: str
    reason: str
    llm_prompt: str | None = None


def serialize_tree(node: TreeNode) -> dict[str, Any]:
    if isinstance(node, LeafNode):
        return {"type": "leaf", "expert_idx": int(node.expert_idx)}
    return {
        "type": "condition",
        "feature": node.spec.feature,
        "comparator": node.spec.comparator,
        "threshold": float(node.spec.threshold),
        "invert": bool(node.spec.invert),
        "if_true": serialize_tree(node.if_true),
        "if_false": serialize_tree(node.if_false),
    }


def deserialize_tree(raw: dict[str, Any]) -> TreeNode:
    if raw["type"] == "leaf":
        return LeafNode(int(raw["expert_idx"]))
    return ConditionNode(
        spec=ConditionSpec(
            feature=str(raw["feature"]),
            comparator=str(raw["comparator"]),
            threshold=float(raw["threshold"]),
            invert=bool(raw.get("invert", False)),
        ),
        if_true=deserialize_tree(raw["if_true"]),
        if_false=deserialize_tree(raw["if_false"]),
    )


def tree_key(node: TreeNode) -> str:
    return json.dumps(serialize_tree(node), sort_keys=True, ensure_ascii=False)


def tree_depth(node: TreeNode) -> int:
    if isinstance(node, LeafNode):
        return 0
    return 1 + max(tree_depth(node.if_true), tree_depth(node.if_false))


def tree_size(node: TreeNode) -> int:
    if isinstance(node, LeafNode):
        return 1
    return 1 + tree_size(node.if_true) + tree_size(node.if_false)


def collect_leaves(node: TreeNode, out: list[int] | None = None) -> list[int]:
    if out is None:
        out = []
    if isinstance(node, LeafNode):
        out.append(int(node.expert_idx))
        return out
    collect_leaves(node.if_true, out)
    collect_leaves(node.if_false, out)
    return out


def collect_specs(node: TreeNode, out: list[ConditionSpec] | None = None) -> list[ConditionSpec]:
    if out is None:
        out = []
    if isinstance(node, LeafNode):
        return out
    out.append(node.spec)
    collect_specs(node.if_true, out)
    collect_specs(node.if_false, out)
    return out


def condition_to_text(spec: ConditionSpec) -> str:
    clause = f"{spec.feature} {spec.comparator} {spec.threshold:.4f}"
    return f"NOT({clause})" if spec.invert else clause


def build_llm_prompt(node: TreeNode, expert_pool: list[dict[str, Any]]) -> str:
    lines = [
        "다음 트레이딩 트리의 경제적 타당성을 점검하라.",
        "목표: 자기모순, 의미 없는 중복 조건, 동일 branch, leaf 편중, 해석 불가능한 조합을 걸러내라.",
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


def heuristic_semantic_filter(node: TreeNode, expert_pool: list[dict[str, Any]], max_depth: int) -> FilterDecision:
    leaves = collect_leaves(node)
    specs = collect_specs(node)
    prompt = build_llm_prompt(node, expert_pool)

    if tree_depth(node) > max_depth:
        return FilterDecision(False, "heuristic", "tree_depth_exceeded", prompt)
    if not specs:
        return FilterDecision(False, "heuristic", "no_conditions", prompt)
    if len(set(leaves)) < 2:
        return FilterDecision(False, "heuristic", "single_expert_only", prompt)
    if tree_size(node) > 15:
        return FilterDecision(False, "heuristic", "tree_too_large", prompt)

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

    if len(leaves) >= 4:
        dominant = max(leaves.count(value) for value in set(leaves)) / len(leaves)
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


def random_leaf(rng: random.Random, expert_count: int) -> LeafNode:
    return LeafNode(expert_idx=rng.randrange(expert_count))


def random_tree(
    rng: random.Random,
    condition_options: list[ConditionSpec],
    expert_count: int,
    max_depth: int,
    depth: int = 0,
) -> TreeNode:
    if depth >= max_depth or (depth > 0 and rng.random() < 0.35):
        return random_leaf(rng, expert_count)
    spec = copy.deepcopy(rng.choice(condition_options))
    return ConditionNode(
        spec=spec,
        if_true=random_tree(rng, condition_options, expert_count, max_depth, depth + 1),
        if_false=random_tree(rng, condition_options, expert_count, max_depth, depth + 1),
    )


def subtree_paths(node: TreeNode, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], TreeNode]]:
    out = [(path, node)]
    if isinstance(node, ConditionNode):
        out.extend(subtree_paths(node.if_true, path + ("if_true",)))
        out.extend(subtree_paths(node.if_false, path + ("if_false",)))
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


def mutate_tree(
    node: TreeNode,
    rng: random.Random,
    condition_options: list[ConditionSpec],
    expert_count: int,
    max_depth: int,
) -> TreeNode:
    target_path, target_node = rng.choice(subtree_paths(node))
    mode = rng.choice(("replace_subtree", "tweak_condition", "tweak_leaf"))
    mutated = copy.deepcopy(node)
    if mode == "replace_subtree":
        return set_subtree(mutated, target_path, random_tree(rng, condition_options, expert_count, max_depth))
    if isinstance(target_node, ConditionNode) and mode == "tweak_condition":
        replacement = copy.deepcopy(target_node)
        replacement.spec = copy.deepcopy(rng.choice(condition_options))
        return set_subtree(mutated, target_path, replacement)
    if isinstance(target_node, LeafNode):
        return set_subtree(mutated, target_path, LeafNode(rng.randrange(expert_count)))
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


def evaluate_tree_codes(node: TreeNode, features: dict[str, np.ndarray]) -> np.ndarray:
    first = next(iter(features.values()))
    out = np.zeros(len(first), dtype="int16")

    def fill(current: TreeNode, mask: np.ndarray) -> None:
        if not np.any(mask):
            return
        if isinstance(current, LeafNode):
            out[mask] = int(current.expert_idx)
            return
        cond = feature_condition_values(current.spec, features)
        fill(current.if_true, mask & cond)
        fill(current.if_false, mask & (~cond))

    fill(node, np.ones(len(first), dtype=bool))
    return out
