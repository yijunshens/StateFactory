
TEXTUAL = "<world state description>"
TEXTUAL_DES = """
State: Every element must be a grammatically complete English sentence that states only observable, factual, and specific information—no speculation or abstraction. Because the representation lacks hierarchical grouping, each sentence must explicitly name its subject to ensure clarity and self-containment.
"""

OBJ_CENTRIC = [
  {
    "object": {
      "<object description>": [
        "<world state description>",
        "<world state description>"
      ]
    }
  },
  {
    "object": {
      "<object description>": [
        "<world state description>"
      ]
    }
  }
]

OBJ_CENTRIC_DES = """
Object: Refers to specific or abstract entities identified in the observations. Objects must be directly grounded in the observed input and should not include inferred or hypothetical elements.
State: A list of distinct attributes or facts associated with the identified object. Each item in the list must NOT assume the key is part of the sentence. It must be a standalone, grammatically complete English sentence that includes its own subject. Unlike the detailed format, this groups multiple observations under the object without requiring specific label keys for each observation.
"""

OBJ_ATTRIBUTE = [
  {
    "object": {
      "<object description>": [
        { "<state name>": "<world state description>" },
        { "<state name>": "<world state description>" }
      ]
    }
  },
  {
    "object": {
      "<object description>": [
        { "<state name>": "<world state description>" }
      ]
    }
  }
]

OBJ_ATTRIBUTE_DES = """
Object: Refers to specific or abstract entities identified in the observations. Objects must be directly grounded in the observed input and should not include inferred or hypothetical elements. Only entities that are explicitly present or clearly implied in the observation should be included.
State: Use key-value pairs to describe object attributes, where the "value" (<world state description>) must NOT assume the key is part of the sentence. It must be a complete, grammatically correct English sentence that explicitly includes a subject (e.g., "The device", "A person", "This object") and a verb, can stand alone as a factual statement, and provides detailed, accurate, and observable facts (not speculation). The "key" (<state name>) must be short and summative.
"""
