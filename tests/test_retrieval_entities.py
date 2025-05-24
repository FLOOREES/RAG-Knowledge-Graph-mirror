from context import benchmarking
from benchmarking.wrappers import CustomWrapper
from benchmarking.general_utils import extract_triplets
from dotenv import load_dotenv
import os

load_dotenv()
client = CustomWrapper()

question = "Who is Luca Maestri and what is his role in the company?"

graph = [
    {
        "relation": "DISCOVERY",
        "label": "discovered the function of",
        "metadata": {},
        "from": {"value": "Dr. Alice Carter", "type": "person", "group": "RESEARCHER"},
        "to": {"value": "protein X", "type": "concept", "group": "BIOLOGY"}
    },
    {
        "relation": "DATE",
        "label": "in the year",
        "metadata": {},
        "from": {"value": "Dr. Alice Carter", "type": "person", "group": "RESEARCHER"},
        "to": {"value": "2019", "type": "date", "group": "YEAR"}
    },
    {
        "relation": "WORKED_ON",
        "label": "conducted research at",
        "metadata": {},
        "from": {"value": "Dr. Alice Carter", "type": "person", "group": "RESEARCHER"},
        "to": {"value": "Cambridge Institute of Genetics", "type": "organization", "group": "UNIVERSITY"}
    },
    {
        "relation": "IMPACT",
        "label": "led to treatment for",
        "metadata": {},
        "from": {"value": "protein X", "type": "concept", "group": "BIOLOGY"},
        "to": {"value": "rare blood disease", "type": "condition", "group": "MEDICINE"}
    },
    {
        "relation": "RELATED_TO",
        "label": "similar to mechanism of",
        "metadata": {},
        "from": {"value": "protein X", "type": "concept", "group": "BIOLOGY"},
        "to": {"value": "protein Y", "type": "concept", "group": "BIOLOGY"}
    },
    {
        "relation": "DIAGNOSED_WITH",
        "label": "was diagnosed with",
        "metadata": {},
        "from": {"value": "Patient 042", "type": "entity", "group": "PATIENT"},
        "to": {"value": "rare blood disease", "type": "condition", "group": "MEDICINE"}
    },
    {
        "relation": "DISCOVERY",
        "label": "discovered mutation in",
        "metadata": {},
        "from": {"value": "Dr. Miguel Torres", "type": "person", "group": "RESEARCHER"},
        "to": {"value": "gene ABC1", "type": "concept", "group": "GENETICS"}
    },
    {
        "relation": "IMPACT",
        "label": "linked to increased risk of",
        "metadata": {},
        "from": {"value": "gene ABC1", "type": "concept", "group": "GENETICS"},
        "to": {"value": "colon cancer", "type": "condition", "group": "MEDICINE"}
    },
    {
        "relation": "WORKED_ON",
        "label": "research conducted at",
        "metadata": {},
        "from": {"value": "Dr. Miguel Torres", "type": "person", "group": "RESEARCHER"},
        "to": {"value": "University of Madrid", "type": "organization", "group": "UNIVERSITY"}
    },
    {
        "relation": "DATE",
        "label": "discovered in",
        "metadata": {},
        "from": {"value": "gene ABC1", "type": "concept", "group": "GENETICS"},
        "to": {"value": "2020", "type": "date", "group": "YEAR"}
    },
    {
        "relation": "RELATED_TO",
        "label": "has similar mutation as",
        "metadata": {},
        "from": {"value": "gene ABC1", "type": "concept", "group": "GENETICS"},
        "to": {"value": "gene XYZ3", "type": "concept", "group": "GENETICS"}
    },
    {
        "relation": "STUDIED_BY",
        "label": "was studied by",
        "metadata": {},
        "from": {"value": "gene XYZ3", "type": "concept", "group": "GENETICS"},
        "to": {"value": "Dr. Nina Kapoor", "type": "person", "group": "RESEARCHER"}
    },
    {
        "relation": "WORKED_ON",
        "label": "conducted clinical trials at",
        "metadata": {},
        "from": {"value": "Dr. Nina Kapoor", "type": "person", "group": "RESEARCHER"},
        "to": {"value": "Stanford Medical Center", "type": "organization", "group": "HOSPITAL"}
    },
    {
        "relation": "TREATED_WITH",
        "label": "treated with",
        "metadata": {},
        "from": {"value": "Patient 077", "type": "entity", "group": "PATIENT"},
        "to": {"value": "experimental therapy B", "type": "treatment", "group": "MEDICINE"}
    },
    {
        "relation": "TARGETS",
        "label": "targets",
        "metadata": {},
        "from": {"value": "experimental therapy B", "type": "treatment", "group": "MEDICINE"},
        "to": {"value": "gene XYZ3", "type": "concept", "group": "GENETICS"}
    }
]

# client.ingest_data(graph)
#
# print(client.input_to_entities(question))
KB_URL = os.getenv('NUCLIA_KB_URL')
print(extract_triplets(KB_URL))