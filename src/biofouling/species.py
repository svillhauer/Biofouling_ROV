"""Display metadata for the four target organisms (class ids live in configs/data.yaml)."""

DISPLAY_NAMES = {
    "rugulopteryx_okamurae": "Rugulopteryx okamurae",
    "asparagopsis_armata": "Asparagopsis armata",
    "botrylloides_violaceus": "Botrylloides violaceus",
    "hildenbrandia_rubra": "Hildenbrandia rubra",
}

# Categorical palette used by every figure, so a species keeps its color
# across plots.
COLORS = {
    "botrylloides_violaceus": "#2a78d6",
    "hildenbrandia_rubra": "#008300",
    "asparagopsis_armata": "#e87ba4",
    "rugulopteryx_okamurae": "#eda100",
}


def display_name(class_name: str, wrap: bool = False) -> str:
    """Human-readable species name; ``wrap`` puts genus and species on two lines."""
    name = DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").capitalize())
    return name.replace(" ", "\n", 1) if wrap else name
