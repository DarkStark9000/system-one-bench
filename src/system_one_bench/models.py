"""The checkpoints under test, pinned to a commit and a weights digest.

The commits are the reviewed pins that the laya package itself ships in
`laya.revisions.PINNED_REVISIONS`. The digests are the SHA-256 of each
`model.safetensors` at that commit, as the Hugging Face API reports it. The
weights at these commits are byte-identical to the Hub's main branch on
2026-10-18, so a result here describes the checkpoint people download today.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Checkpoint:
    name: str
    repo: str
    revision: str
    weights_sha256: str
    encoder: str


CHECKPOINTS: dict[str, Checkpoint] = {
    "english": Checkpoint(
        name="english",
        repo="convaiinnovations/laya",
        revision="55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
        weights_sha256="891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c",
        encoder="ModernBERT-large",
    ),
    "typed-decisions": Checkpoint(
        name="typed-decisions",
        repo="convaiinnovations/laya-typed-decisions",
        revision="1a793eb568e6718f15941d08f85432581df534e3",
        weights_sha256="4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e",
        encoder="ModernBERT-large",
    ),
    "multilingual": Checkpoint(
        name="multilingual",
        repo="convaiinnovations/laya-multilingual",
        revision="e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
        weights_sha256="9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
        encoder="mmBERT-base",
    ),
}


def checkpoint(name: str) -> Checkpoint:
    try:
        return CHECKPOINTS[name]
    except KeyError:
        raise SystemExit(f"unknown checkpoint {name!r}; choose from {', '.join(CHECKPOINTS)}") from None


def check_pins() -> list[str]:
    """Differences between these pins and the ones the installed laya package reviewed."""
    from laya.revisions import PINNED_REVISIONS

    problems = []
    for ckpt in CHECKPOINTS.values():
        pinned = PINNED_REVISIONS.get(ckpt.repo)
        if pinned != ckpt.revision:
            problems.append(f"{ckpt.repo}: bench pins {ckpt.revision}, laya reviewed {pinned}")
    return problems
