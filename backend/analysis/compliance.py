"""Final uniform status - the business rule (config/uniform_config.yaml inference.overall_rule).

any_uniform (default, staff-presence rule):
  >= 1 person inside the store ROI confidently in the approved uniform -> compliant   (green)
  people inside, clothing judged, none in uniform                       -> no_uniform  (red)
  people inside but not enough visual evidence                          -> uncertain   (yellow)
  nobody inside the store ROI                                           -> no_person   (gray)
all_uniform (every judged person must be in uniform):
  any person NON-UNIFORM -> no_uniform; else any UNCERTAIN -> uncertain; else compliant

The system judges clothing only: it never identifies who a person is.
"""
from __future__ import annotations

COMPLIANT, NO_UNIFORM, UNCERTAIN, NO_PERSON = "compliant", "no_uniform", "uncertain", "no_person"

TITLE = {   # ASCII only: also burned into annotated images with OpenCV
    COMPLIANT: "COMPLIANT",
    NO_UNIFORM: "NON-COMPLIANT",
    UNCERTAIN: "UNCERTAIN - PLEASE VERIFY",
    NO_PERSON: "NO PERSON DETECTED",
}
COLOR = {COMPLIANT: "green", NO_UNIFORM: "red", UNCERTAIN: "yellow", NO_PERSON: "gray"}


def decide(labels: list[str], rule: str = "any_uniform") -> str:
    if not labels:
        return NO_PERSON
    if rule == "all_uniform":
        if "non_uniform" in labels:
            return NO_UNIFORM
        return UNCERTAIN if "uncertain" in labels else COMPLIANT
    if "uniform" in labels:
        return COMPLIANT
    if "non_uniform" in labels:
        return NO_UNIFORM
    return UNCERTAIN


def message(status: str, n_uniform: int, n_people: int) -> str:
    if status == COMPLIANT:
        return f"{n_uniform} uniformed person{'s' if n_uniform != 1 else ''} detected inside the selected store area."
    if status == NO_UNIFORM:
        if n_uniform:   # all_uniform rule
            return "At least one person inside the selected store area is not wearing the approved uniform."
        return "No uniformed person detected inside the selected store area."
    if status == UNCERTAIN:
        return "People are visible, but their clothing could not be judged reliably (too small, hidden, blurred or low confidence)."
    return "No person was detected inside the selected store area."
