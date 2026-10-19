"""Compile a minimal resource dependency graph from public task evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from lh_platform.benchmarks.terminal_resource_driver import ResourceSpec
from lh_platform.test_time_adaptation.portable_runtime_kernel import (
    DependencyRole,
    EvidenceState,
    OutputRequirement,
    PortableRuntimeKernel,
    ResourceKind,
    ResourceObservation,
    TaskNode,
    TaskNodeSource,
)

TASK_CONTRACT_COMPILER_V3 = "public_instruction_resource_evidence_v3"
TASK_CONTRACT_COMPILER_V4 = "public_instruction_resource_evidence_v4"
SUPPORTED_TASK_CONTRACT_COMPILERS = frozenset(
    {TASK_CONTRACT_COMPILER_V3, TASK_CONTRACT_COMPILER_V4}
)
_OUTPUT_CUE = re.compile(
    r"\b(create|write|save|generate|produce|move|copy|clone|export|output|"
    r"update|overwrite|edit)\b",
    re.IGNORECASE,
)
_DIRECT_OUTPUT_NEGATION = re.compile(
    r"\b(?:do\s+not|don't|never|must\s+not|mustn't|should\s+not|shouldn't|"
    r"shall\s+not|cannot|can't)\s+$",
    re.IGNORECASE,
)
_CLAUSE_NEGATION = re.compile(
    r"\b(?:do\s+not|don't|never|must\s+not|mustn't|should\s+not|shouldn't|"
    r"shall\s+not|cannot|can't)\b",
    re.IGNORECASE,
)
_PATH_ROLE_CUE = re.compile(
    r"\b(?:(?P<input>found\s+in|read\s+from|extract(?:ed)?\s+from|"
    r"contain(?:ed)?\s+in|source(?:d)?\s+from|process(?:ed)?\s+by|from)|"
    r"(?P<output>called|named|as|to))\b",
    re.IGNORECASE,
)
_KIND_CUES = {
    ResourceKind.DATABASE: ("database",),
    ResourceKind.DATABASE_SIDECAR: (
        "wal file",
        "write-ahead log",
        "write ahead log",
        "write-ahead logging",
    ),
}


@dataclass(frozen=True)
class PublicDependencyEvidence:
    """One deterministic dependency claim derived only from public task text."""

    evidence_id: str
    task_id: str
    resource_id: str
    proposition: str


def _contains_reference(instruction: str, reference: str) -> bool:
    """Match a public path or basename without accepting filename prefixes."""

    pattern = (
        rf"(?<![a-z0-9_.-]){re.escape(reference.casefold())}"
        r"(?!(?:[a-z0-9_-]|\.[a-z0-9]))"
    )
    return re.search(pattern, instruction.casefold()) is not None


def _explicit_public_dependencies(
    *,
    instruction: str,
    resources: Sequence[ResourceSpec],
    initial_observations: Sequence[ResourceObservation],
) -> frozenset[str]:
    """Extract only exact or unambiguous resource-type references from public text."""

    existing = {
        item.resource_id for item in initial_observations if item.observed and item.exists is True
    }
    selected = {
        resource.resource_id
        for resource in resources
        if resource.resource_id in existing
        and (
            _contains_reference(instruction, resource.resource_id)
            or _contains_reference(instruction, PurePosixPath(resource.resource_id).name)
        )
    }
    for kind, cues in _KIND_CUES.items():
        candidates = [
            resource.resource_id
            for resource in resources
            if resource.kind is kind and resource.resource_id in existing
        ]
        if len(candidates) == 1 and any(_contains_reference(instruction, cue) for cue in cues):
            selected.add(candidates[0])
    return frozenset(selected)


def _explicit_public_outputs(
    *,
    instruction: str,
    resources: Sequence[ResourceSpec],
    compiler_id: str,
) -> frozenset[str]:
    """Compile path obligations from public wording, independent of initial state.

    The public contract stays fixed when a stale or partial output already exists.
    Sentence, line, and semicolon boundaries prevent an earlier production cue
    from leaking into a later prohibition or input reference.
    """

    selected: set[str] = set()
    lowered = instruction.casefold()
    for resource in resources:
        relative_reference = resource.resource_id.removeprefix("/app/").casefold()
        references = {
            resource.resource_id.casefold(),
            relative_reference,
            PurePosixPath(resource.resource_id).name.casefold(),
        }
        for reference in references:
            left_boundary = r"(?<![a-z0-9_./-])" if "/" not in reference else r"(?<![a-z0-9_.-])"
            pattern = re.compile(
                rf"{left_boundary}{re.escape(reference)}"
                r"(?!(?:[a-z0-9_-]|\.[a-z0-9]))"
            )
            for match in pattern.finditer(lowered):
                prefix = instruction[: match.start()]
                boundaries = tuple(
                    boundary.end()
                    for boundary in re.finditer(r"(?:^|[\n!?;]|[.](?=\s+[A-Z]))", prefix)
                )
                clause_start = boundaries[-1] if boundaries else 0
                clause_prefix = lowered[clause_start : match.start()]
                output_cues = tuple(_OUTPUT_CUE.finditer(clause_prefix))
                if not output_cues:
                    continue
                # The closest production verb owns the path mention. This keeps a
                # prohibition such as `do not create X` out of the fixed contract.
                closest_cue = output_cues[-1]
                cue_prefix = clause_prefix[: closest_cue.start()]
                if _DIRECT_OUTPUT_NEGATION.search(cue_prefix):
                    continue
                if _CLAUSE_NEGATION.search(cue_prefix):
                    raise ValueError(
                        f"public output contract has ambiguous negation near {resource.resource_id}"
                    )
                # The last explicit relation after the production verb owns the
                # path. This distinguishes `create Y from X` and nested source
                # phrases such as `create Y with the word found in X`.
                cue_to_path = clause_prefix[closest_cue.end() :]
                role_cues = tuple(_PATH_ROLE_CUE.finditer(cue_to_path))
                if role_cues and role_cues[-1].lastgroup == "input":
                    continue
                if compiler_id == TASK_CONTRACT_COMPILER_V4 and not role_cues:
                    prior_mentions: list[tuple[int, int, str]] = []
                    for candidate in resources:
                        candidate_references = {
                            candidate.resource_id.casefold(),
                            candidate.resource_id.removeprefix("/app/").casefold(),
                            PurePosixPath(candidate.resource_id).name.casefold(),
                        }
                        for candidate_reference in candidate_references:
                            candidate_left = (
                                r"(?<![a-z0-9_./-])"
                                if "/" not in candidate_reference
                                else r"(?<![a-z0-9_.-])"
                            )
                            candidate_pattern = re.compile(
                                rf"{candidate_left}{re.escape(candidate_reference)}"
                                r"(?!(?:[a-z0-9_-]|\.[a-z0-9]))"
                            )
                            prior_mentions.extend(
                                (item.start(), item.end(), candidate.resource_id)
                                for item in candidate_pattern.finditer(cue_to_path)
                            )
                    distinct_prior = [
                        item for item in prior_mentions if item[2] != resource.resource_id
                    ]
                    if distinct_prior:
                        previous = max(distinct_prior, key=lambda item: item[1])
                        separator = cue_to_path[previous[1] :]
                        directly_coordinated = re.fullmatch(
                            r"[\s,'\"`]*(?:and|or)[\s,'\"`]*(?:the\s+file\s+)?",
                            separator,
                            re.IGNORECASE,
                        )
                        if directly_coordinated is None:
                            continue
                selected.add(resource.resource_id)
                break
            if resource.resource_id in selected:
                break
    return frozenset(selected)


def _public_output_groups(
    *, instruction: str, resource_ids: Sequence[str]
) -> tuple[tuple[str, ...], ...]:
    """Group outputs only when their path mentions are directly joined by `or`."""

    lowered = instruction.casefold()
    remaining = list(resource_ids)
    groups: list[tuple[str, ...]] = []
    while remaining:
        anchor = remaining.pop(0)
        group = [anchor]
        anchor_name = re.escape(PurePosixPath(anchor).name.casefold())
        for candidate in tuple(remaining):
            candidate_name = re.escape(PurePosixPath(candidate).name.casefold())
            separator = r"[\s'\"`]*\bor\b[\s'\"`]*(?:/app/)?"
            alternative = re.search(
                rf"{anchor_name}{separator}{candidate_name}"
                rf"|{candidate_name}{separator}{anchor_name}",
                lowered,
            )
            if alternative is not None:
                group.append(candidate)
                remaining.remove(candidate)
        groups.append(tuple(group))
    return tuple(groups)


def compile_initial_resource_task_graph(
    *,
    instruction: str,
    resources: Sequence[ResourceSpec],
    initial_observations: Sequence[ResourceObservation],
    compiler_id: str = TASK_CONTRACT_COMPILER_V3,
) -> tuple[TaskNode, ...]:
    """Separate the fixed public goal from revisable initial-input assumptions.

    An actor-visible resource that exists before the first actor action is treated
    as an inferred prerequisite. A missing path becomes a fixed output obligation
    only when the public instruction names it near an explicit production verb.
    Later existence advances that child to awaiting evaluation; it never proves
    semantic correctness by itself.
    """

    if compiler_id not in SUPPORTED_TASK_CONTRACT_COMPILERS:
        raise ValueError("task contract compiler identity is unsupported")
    if not instruction.strip():
        raise ValueError("public task instruction is required")
    observations = {item.resource_id: item for item in initial_observations}
    if len(observations) != len(initial_observations):
        raise ValueError("initial resource observations must be unique")
    resource_ids = {item.resource_id for item in resources}
    if set(observations) != resource_ids:
        raise ValueError("resource declarations and initial observations differ")
    nodes = [
        TaskNode(
            task_id="public-task",
            parent_id=None,
            title=instruction.strip(),
            public_requirement=True,
            source=TaskNodeSource.PUBLIC_CONTRACT,
        )
    ]
    canonical_resources = tuple(sorted(resources, key=lambda item: item.resource_id))
    public_outputs = _explicit_public_outputs(
        instruction=instruction,
        resources=canonical_resources,
        compiler_id=compiler_id,
    )
    output_groups = _public_output_groups(
        instruction=instruction,
        resource_ids=tuple(
            resource.resource_id
            for resource in canonical_resources
            if resource.resource_id in public_outputs
        ),
    )
    output_group_by_resource = {
        resource_id: group for group in output_groups for resource_id in group
    }
    for index, resource in enumerate(canonical_resources):
        observation = observations[resource.resource_id]
        if resource.resource_id in public_outputs:
            group = output_group_by_resource[resource.resource_id]
            if resource.resource_id == group[0]:
                nodes.append(
                    TaskNode(
                        task_id=f"public-output-{index:03d}",
                        parent_id="public-task",
                        title="Produce the public output: " + " or ".join(group),
                        required_output_resource_ids=group,
                        output_requirement=(
                            OutputRequirement.ANY if len(group) > 1 else OutputRequirement.ALL
                        ),
                        public_requirement=True,
                        source=TaskNodeSource.PUBLIC_CONTRACT,
                    )
                )
        if not observation.observed or observation.exists is not True:
            continue
        nodes.append(
            TaskNode(
                task_id=f"resource-prerequisite-{index:03d}",
                parent_id="public-task",
                title=f"Validate whether the task still needs: {resource.resource_id}",
                hypothesized_prerequisite_resource_ids=(resource.resource_id,),
                public_requirement=False,
                source=TaskNodeSource.INFERRED_PLAN,
            )
        )
    return tuple(nodes)


def apply_public_instruction_dependency_evidence(
    *,
    kernel: PortableRuntimeKernel,
    instruction: str,
    resources: Sequence[ResourceSpec],
    initial_observations: Sequence[ResourceObservation],
    compiler_id: str = TASK_CONTRACT_COMPILER_V3,
) -> tuple[str, ...]:
    """Promote public, unambiguous input references through durable evidence events.

    This locally designed compiler does not infer file purpose or a repair. It only
    promotes an existing-resource hypothesis when the public instruction names the
    exact resource or uniquely identifies its declared resource type.
    """

    compiled = compile_public_instruction_dependency_evidence(
        instruction=instruction,
        resources=resources,
        initial_observations=initial_observations,
        compiler_id=compiler_id,
    )
    evidence_ids: list[str] = []
    for item in compiled:
        kernel.record_evidence(
            evidence_id=item.evidence_id,
            task_id=item.task_id,
            proposition=item.proposition,
            state=EvidenceState.SUPPORTED,
            resource_ids=(),
            dependency_claims={item.resource_id: DependencyRole.CONFIRMED},
            dependency_claims_version_bound=False,
        )
        kernel.revise_inferred_task_prerequisites(
            task_id=item.task_id,
            prerequisite_resource_ids=(item.resource_id,),
            hypothesized_prerequisite_resource_ids=(),
            evidence_id=item.evidence_id,
        )
        evidence_ids.append(item.evidence_id)
    return tuple(evidence_ids)


def compile_public_instruction_dependency_evidence(
    *,
    instruction: str,
    resources: Sequence[ResourceSpec],
    initial_observations: Sequence[ResourceObservation],
    compiler_id: str = TASK_CONTRACT_COMPILER_V3,
) -> tuple[PublicDependencyEvidence, ...]:
    """Return the independently replayable public dependency claims."""

    if compiler_id not in SUPPORTED_TASK_CONTRACT_COMPILERS:
        raise ValueError("task contract compiler identity is unsupported")

    explicit = _explicit_public_dependencies(
        instruction=instruction,
        resources=resources,
        initial_observations=initial_observations,
    )
    compiled: list[PublicDependencyEvidence] = []
    # Use the same canonical enumeration as graph construction so an evidence
    # record cannot point at a different child when config order changes.
    for index, resource in enumerate(sorted(resources, key=lambda item: item.resource_id)):
        if resource.resource_id not in explicit:
            continue
        task_id = f"resource-prerequisite-{index:03d}"
        compiled.append(
            PublicDependencyEvidence(
                evidence_id=f"{compiler_id}:{task_id}",
                task_id=task_id,
                resource_id=resource.resource_id,
                proposition=(
                    "The public instruction explicitly identifies this resource "
                    f"as task input: {resource.resource_id}"
                ),
            )
        )
    return tuple(compiled)


__all__ = [
    "SUPPORTED_TASK_CONTRACT_COMPILERS",
    "TASK_CONTRACT_COMPILER_V3",
    "TASK_CONTRACT_COMPILER_V4",
    "apply_public_instruction_dependency_evidence",
    "compile_public_instruction_dependency_evidence",
    "compile_initial_resource_task_graph",
]
