"""家族树数据：两棵同构家族树、全部查询及其 multi-hot 目标"""

from collections import defaultdict

import numpy as np

PEOPLE = [
    "Christopher", "Penelope", "Andrew", "Christine",
    "Margaret", "Arthur", "Victoria", "James",
    "Jennifer", "Charles", "Colin", "Charlotte",
    "Roberto", "Maria", "Piero", "Francesca",
    "Gina", "Emilio", "Lucia", "Marco",
    "Angela", "Tomaso", "Alfonso", "Sophia",
]
RELATIONS = [
    "father", "mother", "husband", "wife",
    "son", "daughter", "brother", "sister",
    "uncle", "aunt", "nephew", "niece",
]
MALE = {
    "Christopher", "Andrew", "Arthur", "James", "Charles", "Colin",
    "Roberto", "Piero", "Emilio", "Marco", "Tomaso", "Alfonso",
}
GENDER = {person: ("M" if person in MALE else "F") for person in PEOPLE}
COUPLES = [
    ("Christopher", "Penelope"), ("Andrew", "Christine"),
    ("Arthur", "Margaret"), ("James", "Victoria"),
    ("Charles", "Jennifer"), ("Roberto", "Maria"),
    ("Piero", "Francesca"), ("Emilio", "Gina"),
    ("Marco", "Lucia"), ("Tomaso", "Angela"),
]
PARENT_GROUPS = [  # (父亲, 母亲, 子女)
    ("Christopher", "Penelope", ["Arthur", "Victoria"]),
    ("Andrew", "Christine", ["James", "Jennifer"]),
    ("James", "Victoria", ["Colin", "Charlotte"]),
    ("Roberto", "Maria", ["Emilio", "Lucia"]),
    ("Piero", "Francesca", ["Marco", "Angela"]),
    ("Marco", "Lucia", ["Alfonso", "Sophia"]),
]
HOLDOUT_QUERIES = [  # 留出集：训练时完全不出现
    ("Colin", "mother"),
    ("Charlotte", "father"),
    ("Alfonso", "mother"),
    ("Sophia", "father"),
]

person_index = {person: i for i, person in enumerate(PEOPLE)}
relation_index = {relation: i for i, relation in enumerate(RELATIONS)}


def build_facts():
    """由婚姻与亲子关系推出全部 (人物, 关系, 答案) 三元组，共 112 条。"""
    spouse_of = {}
    for husband, wife in COUPLES:
        spouse_of[husband] = wife
        spouse_of[wife] = husband

    parents_of = {}
    siblings_of = defaultdict(set)
    for father, mother, children in PARENT_GROUPS:
        for child in children:
            parents_of[child] = (father, mother)
            siblings_of[child].update(other for other in children if other != child)

    facts = set()
    for husband, wife in COUPLES:
        facts.update({(wife, "husband", husband), (husband, "wife", wife)})

    for father, mother, children in PARENT_GROUPS:
        for child in children:
            facts.update({(child, "father", father), (child, "mother", mother)})
            relation = "son" if GENDER[child] == "M" else "daughter"
            facts.update({(father, relation, child), (mother, relation, child)})

    for person, siblings in siblings_of.items():
        for sibling in siblings:
            relation = "brother" if GENDER[sibling] == "M" else "sister"
            facts.add((person, relation, sibling))

    # 父母的兄弟姐妹及其配偶是 uncle / aunt
    for child, parents in parents_of.items():
        for parent in parents:
            for blood_relative in siblings_of[parent]:
                for relative in (blood_relative, spouse_of[blood_relative]):
                    relation = "uncle" if GENDER[relative] == "M" else "aunt"
                    inverse = "nephew" if GENDER[child] == "M" else "niece"
                    facts.update({(child, relation, relative), (relative, inverse, child)})
    return facts


def build_dataset():
    """把查询 (人物, 关系) 编码为两个 one-hot 输入，答案编码为 24 维 multi-hot 目标。"""
    facts = build_facts()
    answers_by_query = defaultdict(set)
    for person, relation, answer in facts:
        answers_by_query[(person, relation)].add(answer)

    cases = sorted(
        answers_by_query,
        key=lambda query: (person_index[query[0]], relation_index[query[1]]),
    )
    person_inputs = np.zeros((len(cases), len(PEOPLE)))
    relation_inputs = np.zeros((len(cases), len(RELATIONS)))
    targets = np.zeros((len(cases), len(PEOPLE)))
    for row, (person, relation) in enumerate(cases):
        person_inputs[row, person_index[person]] = 1.0
        relation_inputs[row, relation_index[relation]] = 1.0
        for answer in answers_by_query[(person, relation)]:
            targets[row, person_index[answer]] = 1.0

    holdout_indices = np.array([cases.index(query) for query in HOLDOUT_QUERIES])
    holdout_set = set(holdout_indices.tolist())
    train_indices = np.array([i for i in range(len(cases)) if i not in holdout_set])

    assert len(facts) == 112
    assert len(cases) == 104
    assert len(train_indices) == 100 and len(holdout_indices) == 4
    return person_inputs, relation_inputs, targets, train_indices, holdout_indices, answers_by_query


def encode_queries(queries):
    """把 (人物, 关系) 查询列表编码为 one-hot 输入，返回形状 (n, 24) 与 (n, 12) 的数组。"""
    person_inputs = np.zeros((len(queries), len(PEOPLE)))
    relation_inputs = np.zeros((len(queries), len(RELATIONS)))
    for row, (person, relation) in enumerate(queries):
        if person not in person_index:
            raise ValueError(f"未知人物: {person}")
        if relation not in relation_index:
            raise ValueError(f"未知关系: {relation}")
        person_inputs[row, person_index[person]] = 1.0
        relation_inputs[row, relation_index[relation]] = 1.0
    return person_inputs, relation_inputs
