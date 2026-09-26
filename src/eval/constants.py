"""
Fixed facts the evaluation harness relies on: speakers, severity, mics, vocabularies.
"""

from src.config import Config

COMMANDS = tuple(Config.TARGET_COMMANDS)

# Mic folder names as they appear in the `mic` column (src/data/preprocessing.py)
ARRAY_MIC = "wav_arrayMic"
HEAD_MIC = "wav_headMic"
MICS = (ARRAY_MIC, HEAD_MIC)
MIC_LABELS = {ARRAY_MIC: "array mic", HEAD_MIC: "head mic"}
# The array mic sits at a distance, the closer match to an appliance or a phone
# on a table, so it carries the headline. Never average the two mics'
# predictions: the deployed device has one mic.
HEADLINE_MIC = ARRAY_MIC

# Predictions that are not one of the commands
REJECT = "reject"  # the model declined to answer (step 5 reject class)
OOV = "oov"        # open-vocabulary ASR output that maps to no command
NON_COMMAND_PREDS = (REJECT, OOV)

# TORGO severity from its clinical ratings
SEVERITY = {
    "severe": ("F01", "M01", "M02", "M04"),
    "moderate-severe": ("M05",),
    "mild": ("F03", "F04", "M03"),
}
DYSARTHRIC_SPEAKERS = tuple(sorted(s for group in SEVERITY.values() for s in group))
CONTROL_SPEAKERS = ("FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04")

# Google Speech Commands v0.02 vocabulary (35 words), used to split results into
# words the small model saw in pretraining and words it did not
SPEECH_COMMANDS_V2_WORDS = (
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "visual", "wow", "yes", "zero",
)
