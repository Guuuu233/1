"""Unit tests for E-02: Evidence relation reducer folding and contribution cap.

Frozen Contract v4 (DAV-772 / DAV-794):
- SOURCE_REPETITION as undirected equivalence folding (idempotent, no cycles).
- DERIVED_OBSERVATION as directed derivation (cycle detection & terminal analysis).
- Pure repetition components have derived_terminal_ids=().
- Canonical JSON encoding for component_id prevents comma collisions.
- Explicit reuse of E-01 FailClosedReason and ReducerFailReason.
- Zero independent vote inference, global_contribution_cap=1, independence_status=UNKNOWN.
- Strict input permutation and edge reversal invariance.
"""
from __future__ import annotations

import hashlib
import json
import random
import pytest

from tradingagents.agents.utils.claim_cluster import (
    EvidenceReductionError,
    EvidenceReductionResult,
    FoldedComponent,
    IndependenceStatus,
    ReducerFailReason,
    reduce_evidence_claims,
)
from tradingagents.agents.utils.evidence_relations import (
    EvidenceRelation,
    EvidenceRelationGraph,
    FailClosedReason,
    RelationType,
)


class TestRedTeamMinimumCases:
    """Mandatory minimum red team scenarios required by DAV-772 / DAV-794."""

    def test_tc01_repetition_reverse_pair_folds_to_one_component_no_cycle(self):
        """repetition A->B and B->A must fold into 1 component without cycle, cap=1."""
        claim_ids = ["A", "B"]
        rel_ab = EvidenceRelation(
            source_id="A",
            relation_type=RelationType.SOURCE_REPETITION,
            target_id="B",
        )
        rel_ba = EvidenceRelation(
            source_id="B",
            relation_type=RelationType.SOURCE_REPETITION,
            target_id="A",
        )

        # Both orderings and duplicate presence
        res = reduce_evidence_claims(claim_ids, [rel_ab, rel_ba])

        assert len(res.folded_components) == 1
        comp = res.folded_components[0]
        assert comp.member_claim_ids == ("A", "B")
        assert comp.derived_terminal_ids == ()
        assert len(comp.audit_edges) == 1
        assert comp.audit_edges[0].source_id == "A"
        assert comp.audit_edges[0].target_id == "B"
        assert comp.audit_edges[0].relation_type == RelationType.SOURCE_REPETITION

        assert res.unconnected_claim_ids == ()
        assert res.independence_status == IndependenceStatus.UNKNOWN
        assert res.global_contribution_cap == 1

    def test_tc02_repetition_triangle_folds_to_one_component_no_cycle(self):
        """repetition A-B / B-C / C-A must fold into 1 component without cycle, cap=1."""
        claim_ids = ["A", "B", "C"]
        r1 = EvidenceRelation("A", RelationType.SOURCE_REPETITION, "B")
        r2 = EvidenceRelation("B", RelationType.SOURCE_REPETITION, "C")
        r3 = EvidenceRelation("C", RelationType.SOURCE_REPETITION, "A")

        res = reduce_evidence_claims(claim_ids, [r1, r2, r3])

        assert len(res.folded_components) == 1
        comp = res.folded_components[0]
        assert comp.member_claim_ids == ("A", "B", "C")
        assert comp.derived_terminal_ids == ()
        assert len(comp.audit_edges) == 3
        assert res.unconnected_claim_ids == ()
        assert res.global_contribution_cap == 1
        assert res.independence_status == IndependenceStatus.UNKNOWN

    def test_tc03_derived_directed_cycle_detected_fails_closed(self):
        """derived A->B and B->A must fail-closed as CYCLE_DETECTED."""
        claim_ids = ["A", "B"]
        r1 = EvidenceRelation("A", RelationType.DERIVED_OBSERVATION, "B")
        r2 = EvidenceRelation("B", RelationType.DERIVED_OBSERVATION, "A")

        with pytest.raises(EvidenceReductionError) as exc_info:
            reduce_evidence_claims(claim_ids, [r1, r2])

        err = exc_info.value
        assert err.error_reason == FailClosedReason.CYCLE_DETECTED
        assert "CYCLE_DETECTED" in str(err)
        assert "A" in err.affected_nodes
        assert "B" in err.affected_nodes

    def test_tc04_pure_repetition_has_empty_derived_terminals(self):
        """pure repetition A-B has derived_terminal_ids=() and ambiguous terminal flag."""
        claim_ids = ["A", "B"]
        r1 = EvidenceRelation("A", RelationType.SOURCE_REPETITION, "B")

        res = reduce_evidence_claims(claim_ids, [r1])

        assert len(res.folded_components) == 1
        comp = res.folded_components[0]
        assert comp.member_claim_ids == ("A", "B")
        assert comp.derived_terminal_ids == ()
        assert comp.is_ambiguous_derived_terminal is True

        with pytest.raises(EvidenceReductionError) as exc_info:
            comp.get_single_derived_terminal()

        err = exc_info.value
        assert err.error_reason == ReducerFailReason.AMBIGUOUS_DERIVED_TERMINAL

    def test_tc05_repetition_plus_derived_hybrid_single_terminal(self):
        """repetition A-B + derived B->C forms 1 component with derived_terminal_ids=(C,)."""
        claim_ids = ["A", "B", "C"]
        r1 = EvidenceRelation("A", RelationType.SOURCE_REPETITION, "B")
        r2 = EvidenceRelation("B", RelationType.DERIVED_OBSERVATION, "C")

        res = reduce_evidence_claims(claim_ids, [r1, r2])

        assert len(res.folded_components) == 1
        comp = res.folded_components[0]
        assert comp.member_claim_ids == ("A", "B", "C")
        assert comp.derived_terminal_ids == ("C",)
        assert comp.is_ambiguous_derived_terminal is False
        assert comp.get_single_derived_terminal() == "C"
        assert res.global_contribution_cap == 1

    def test_tc06_comma_claim_ids_no_hash_collision(self):
        """Member sets with commas must have distinct canonical payloads and component_ids."""
        case1 = ["a", "b,c"]
        case2 = ["a,b", "c"]
        case3 = ["a", "b", "c"]

        p1 = json.dumps(case1, ensure_ascii=False, separators=(",", ":"))
        p2 = json.dumps(case2, ensure_ascii=False, separators=(",", ":"))
        p3 = json.dumps(case3, ensure_ascii=False, separators=(",", ":"))

        assert p1 != p2
        assert p2 != p3
        assert p1 != p3

        h1 = hashlib.sha256(p1.encode("utf-8")).hexdigest()[:12]
        h2 = hashlib.sha256(p2.encode("utf-8")).hexdigest()[:12]
        h3 = hashlib.sha256(p3.encode("utf-8")).hexdigest()[:12]

        assert h1 != h2
        assert h2 != h3
        assert h1 != h3

        # Test within reducer execution
        r1 = reduce_evidence_claims(case1, [EvidenceRelation("a", RelationType.SOURCE_REPETITION, "b,c")])
        r2 = reduce_evidence_claims(case2, [EvidenceRelation("a,b", RelationType.SOURCE_REPETITION, "c")])

        id1 = r1.folded_components[0].component_id
        id2 = r2.folded_components[0].component_id
        assert id1 != id2
        assert f"_{h1}" in id1
        assert f"_{h2}" in id2

    def test_tc07_self_loop_fails_closed(self):
        """Self-loop edge must fail-closed as SELF_LOOP."""
        claim_ids = ["A"]
        r1 = EvidenceRelation("A", RelationType.SOURCE_REPETITION, "A")

        with pytest.raises(EvidenceReductionError) as exc_info:
            reduce_evidence_claims(claim_ids, [r1])

        err = exc_info.value
        assert err.error_reason == FailClosedReason.SELF_LOOP
        assert err.affected_nodes == ("A",)

    def test_tc08_dangling_reference_fails_closed(self):
        """Edge referencing node not in claim_ids must fail-closed as DANGLING_REFERENCE."""
        claim_ids = ["A"]
        r1 = EvidenceRelation("A", RelationType.DERIVED_OBSERVATION, "GHOST")

        with pytest.raises(EvidenceReductionError) as exc_info:
            reduce_evidence_claims(claim_ids, [r1])

        err = exc_info.value
        assert err.error_reason == FailClosedReason.DANGLING_REFERENCE
        assert "GHOST" in err.affected_nodes

    def test_tc09_malformed_claim_sequence_fails_closed(self):
        """Malformed claim_ids sequence must fail-closed as MALFORMED_CLAIM_SEQUENCE."""
        invalid_inputs = [
            None,
            "not_a_sequence",
            {"A": 1},
            ["A", ""],
            ["A", "   "],
            ["A", None],
            ["A", 123],
            ["A", {"claim": "bad"}],
            ["A", ["nested"]],
        ]

        for bad_input in invalid_inputs:
            with pytest.raises(EvidenceReductionError) as exc_info:
                reduce_evidence_claims(bad_input, [])  # type: ignore[arg-type]
            assert exc_info.value.error_reason == ReducerFailReason.MALFORMED_CLAIM_SEQUENCE


class TestPermutationAndDeduplicationInvariance:
    """Tests ensuring input ordering, reverse edges, and duplicate edges do not alter output."""

    def test_claim_ids_and_relations_permutation_invariance(self):
        """Permuting claim_ids and relations produces bitwise identical results."""
        claim_ids = ["CLM-01", "CLM-02", "CLM-03", "CLM-04", "CLM-05", "CLM-06"]
        relations = [
            EvidenceRelation("CLM-01", RelationType.SOURCE_REPETITION, "CLM-02"),
            EvidenceRelation("CLM-02", RelationType.DERIVED_OBSERVATION, "CLM-03"),
            EvidenceRelation("CLM-04", RelationType.SOURCE_REPETITION, "CLM-05"),
        ]

        baseline = reduce_evidence_claims(claim_ids, relations)

        rng = random.Random(42)
        for _ in range(20):
            perm_claims = list(claim_ids)
            perm_rels = list(relations)
            rng.shuffle(perm_claims)
            rng.shuffle(perm_rels)

            res = reduce_evidence_claims(perm_claims, perm_rels)
            assert res.folded_components == baseline.folded_components
            assert res.unconnected_claim_ids == baseline.unconnected_claim_ids
            assert res.audit_edges == baseline.audit_edges
            assert res.independence_status == baseline.independence_status
            assert res.global_contribution_cap == baseline.global_contribution_cap

    def test_reverse_repetition_edge_invariance(self):
        """A-REP->B vs B-REP->A vs both must yield identical output and audit edges."""
        claims = ["A", "B"]
        r_ab = EvidenceRelation("A", RelationType.SOURCE_REPETITION, "B")
        r_ba = EvidenceRelation("B", RelationType.SOURCE_REPETITION, "A")

        res_ab = reduce_evidence_claims(claims, [r_ab])
        res_ba = reduce_evidence_claims(claims, [r_ba])
        res_both = reduce_evidence_claims(claims, [r_ab, r_ba])
        res_both_rev = reduce_evidence_claims(claims, [r_ba, r_ab])

        assert res_ab.folded_components == res_ba.folded_components
        assert res_ab.folded_components == res_both.folded_components
        assert res_ab.folded_components == res_both_rev.folded_components
        assert res_ab.audit_edges == res_ba.audit_edges

    def test_duplicate_edges_idempotent_invariance(self):
        """Duplicate identical relations do not change components or audit edges."""
        claims = ["A", "B", "C"]
        r1 = EvidenceRelation("A", RelationType.SOURCE_REPETITION, "B")
        r2 = EvidenceRelation("B", RelationType.DERIVED_OBSERVATION, "C")

        res_single = reduce_evidence_claims(claims, [r1, r2])
        res_multi = reduce_evidence_claims(claims, [r1, r2, r1, r2, r1])

        assert res_single.folded_components == res_multi.folded_components
        assert res_single.audit_edges == res_multi.audit_edges


class TestTopologyAndTerminalResolution:
    """Tests for derived terminal extraction, DAGs, and multi-terminal behaviors."""

    def test_ambiguous_derived_terminals_two_bases(self):
        """When a component has multiple derived terminals, both are stably listed."""
        claims = ["A", "B", "C"]
        # A derives from B, and A derives from C
        r1 = EvidenceRelation("A", RelationType.DERIVED_OBSERVATION, "B")
        r2 = EvidenceRelation("A", RelationType.DERIVED_OBSERVATION, "C")

        res = reduce_evidence_claims(claims, [r1, r2])

        assert len(res.folded_components) == 1
        comp = res.folded_components[0]
        assert comp.member_claim_ids == ("A", "B", "C")
        assert comp.derived_terminal_ids == ("B", "C")
        assert comp.is_ambiguous_derived_terminal is True

        with pytest.raises(EvidenceReductionError) as exc_info:
            comp.get_single_derived_terminal()
        assert exc_info.value.error_reason == ReducerFailReason.AMBIGUOUS_DERIVED_TERMINAL

    def test_derived_chain_resolves_to_single_terminal(self):
        """Derived observation chain A->B->C resolves to single terminal (C,)."""
        claims = ["A", "B", "C"]
        r1 = EvidenceRelation("A", RelationType.DERIVED_OBSERVATION, "B")
        r2 = EvidenceRelation("B", RelationType.DERIVED_OBSERVATION, "C")

        res = reduce_evidence_claims(claims, [r1, r2])

        assert len(res.folded_components) == 1
        comp = res.folded_components[0]
        assert comp.derived_terminal_ids == ("C",)
        assert comp.get_single_derived_terminal() == "C"

    def test_diamond_dag_resolves_to_single_terminal(self):
        """Diamond DAG A->B, A->C, B->D, C->D resolves to terminal (D,)."""
        claims = ["A", "B", "C", "D"]
        relations = [
            EvidenceRelation("A", RelationType.DERIVED_OBSERVATION, "B"),
            EvidenceRelation("A", RelationType.DERIVED_OBSERVATION, "C"),
            EvidenceRelation("B", RelationType.DERIVED_OBSERVATION, "D"),
            EvidenceRelation("C", RelationType.DERIVED_OBSERVATION, "D"),
        ]

        res = reduce_evidence_claims(claims, relations)

        assert len(res.folded_components) == 1
        comp = res.folded_components[0]
        assert comp.derived_terminal_ids == ("D",)
        assert comp.get_single_derived_terminal() == "D"

    def test_longer_directed_cycle_detected(self):
        """Cycle of length 3 in derived observation subgraph is detected."""
        claims = ["A", "B", "C"]
        relations = [
            EvidenceRelation("A", RelationType.DERIVED_OBSERVATION, "B"),
            EvidenceRelation("B", RelationType.DERIVED_OBSERVATION, "C"),
            EvidenceRelation("C", RelationType.DERIVED_OBSERVATION, "A"),
        ]

        with pytest.raises(EvidenceReductionError) as exc_info:
            reduce_evidence_claims(claims, relations)
        assert exc_info.value.error_reason == FailClosedReason.CYCLE_DETECTED


class TestIndependenceAndContributionCapInvariants:
    """Tests ensuring no independent voting leakage and strict cap=1 enforcement."""

    def test_empty_claims_has_cap_zero(self):
        """Zero claims input produces global_contribution_cap=0."""
        res = reduce_evidence_claims([], [])
        assert res.folded_components == ()
        assert res.unconnected_claim_ids == ()
        assert res.audit_edges == ()
        assert res.independence_status == IndependenceStatus.UNKNOWN
        assert res.global_contribution_cap == 0

    def test_all_unconnected_claims_remain_unknown_with_cap_one(self):
        """Claims with no relations remain unconnected; cap is 1, not N."""
        claims = ["C1", "C2", "C3", "C4", "C5"]
        res = reduce_evidence_claims(claims, [])

        assert len(res.folded_components) == 0
        assert res.unconnected_claim_ids == ("C1", "C2", "C3", "C4", "C5")
        assert res.audit_edges == ()
        assert res.independence_status == IndependenceStatus.UNKNOWN
        assert res.global_contribution_cap == 1

    def test_disconnected_folded_components_cap_remains_one(self):
        """Multiple disjoint components + unconnected claims have cap=1 (never sums components)."""
        claims = ["A1", "A2", "B1", "B2", "U1", "U2"]
        relations = [
            EvidenceRelation("A1", RelationType.SOURCE_REPETITION, "A2"),
            EvidenceRelation("B1", RelationType.DERIVED_OBSERVATION, "B2"),
        ]

        res = reduce_evidence_claims(claims, relations)

        assert len(res.folded_components) == 2
        assert res.unconnected_claim_ids == ("U1", "U2")
        assert res.independence_status == IndependenceStatus.UNKNOWN
        assert res.global_contribution_cap == 1

        # Check absence of any voting fields in result dataclass
        assert not hasattr(res, "votes")
        assert not hasattr(res, "effective_independent_votes")
        assert not hasattr(res, "weights")
        assert not hasattr(res, "are_claims_proven_independent")

    def test_unrelated_edge_types_do_not_fold(self):
        """SUPPORTS / REFUTES relations do not fold claims into components."""
        claims = ["A", "B"]
        r_sup = EvidenceRelation("A", RelationType.SUPPORTS, "B")

        res = reduce_evidence_claims(claims, [r_sup])

        assert len(res.folded_components) == 0
        assert res.unconnected_claim_ids == ("A", "B")
        assert res.audit_edges == ()
        assert res.global_contribution_cap == 1


class TestContainerAndTypeRobustness:
    """Tests for graph containers, input types, and error robustness."""

    def test_accepts_evidence_relation_graph_container(self):
        """reduce_evidence_claims accepts EvidenceRelationGraph instance."""
        claims = ["A", "B"]
        graph = EvidenceRelationGraph.from_relations([
            EvidenceRelation("A", RelationType.SOURCE_REPETITION, "B"),
        ])

        res = reduce_evidence_claims(claims, graph)
        assert len(res.folded_components) == 1
        assert res.folded_components[0].member_claim_ids == ("A", "B")

    def test_rejects_non_evidence_relation_elements(self):
        """Non-EvidenceRelation items in relations raise TypeError."""
        with pytest.raises(TypeError, match="must be EvidenceRelation"):
            reduce_evidence_claims(["A", "B"], ["not_a_relation"])  # type: ignore[list-item]

    def test_rejects_non_sequence_relations(self):
        """Non-sequence relations raises TypeError."""
        with pytest.raises(TypeError):
            reduce_evidence_claims(["A", "B"], None)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            reduce_evidence_claims(["A", "B"], 123)  # type: ignore[arg-type]
