"""Final uniform status - the business rule.

  >= 1 person inside the store ROI confidently in the approved uniform -> compliant   (green)
  people inside, clothing judged, none in uniform                       -> no_uniform  (red)
  people inside but not enough visual evidence                          -> uncertain   (yellow)
  nobody inside the store ROI                                           -> no_person   (gray)

The system judges clothing only: it never identifies who a person is.
"""
from __future__ import annotations

COMPLIANT, NO_UNIFORM, UNCERTAIN, NO_PERSON = "compliant", "no_uniform", "uncertain", "no_person"

TITLE = {
    COMPLIANT: "UNIFORM COMPLIANT",
    NO_UNIFORM: "NO UNIFORM DETECTED",
    UNCERTAIN: "UNCERTAIN",
    NO_PERSON: "NO PERSON DETECTED",
}
COLOR = {COMPLIANT: "green", NO_UNIFORM: "red", UNCERTAIN: "yellow", NO_PERSON: "gray"}


def decide(labels: list[str]) -> str:
    if not labels:
        return NO_PERSON
    if "uniform" in labels:
        return COMPLIANT
    if "non_uniform" in labels:
        return NO_UNIFORM
    return UNCERTAIN


def message(status: str, n_uniform: int, n_people: int) -> str:
    if status == COMPLIANT:
        return f"{n_uniform} uniformed person{'s' if n_uniform != 1 else ''} detected inside the selected store area."
    if status == NO_UNIFORM:
        return "No uniformed person detected inside the selected store area."
    if status == UNCERTAIN:
        return "People are visible, but their clothing could not be judged reliably (too small, hidden, blurred or low confidence)."
    return "No person was detected inside the selected store area."
