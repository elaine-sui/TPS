import os

parent_dir = '.'
CONCEPT_EMBED_DIR_TEMPLATE = '{parent_dir}/concept_embeds_{concept_type}'
CLASS_EMBED_DIR = f'{parent_dir}/class_embeds'
COOP_EMBED_DIR = f'{parent_dir}/coop_embeds'
CLASS2CONCEPT_DICT_TEMPLATE = '{parent_dir}/concept_dict_{concept_type}'
PROJ_MATRIX_DIR = f'{parent_dir}/proj_matrix'
CLASS_EMBED_W_TEMPLATES_DIR = f'{parent_dir}/class_embeds_w_imagenet_templates'

os.makedirs(PROJ_MATRIX_DIR, exist_ok=True)

tpt_to_regular_map = {
    'I': 'ImageNet',
    'flower102': 'flower',
    'food101': 'food101',
    'dtd': 'DTD',
    'aircraft': 'aircraft',
    'ucf101': 'UCF101',
    'eurosat': 'EuroSAT',
    'caltech101': 'CalTech101',
    'cars': 'cars',
    'pets': 'pets',
    'sun397': 'SUN397'
}

tpt_to_labo_map = {
    'I': 'ImageNet',
    'flower102': 'flower',
    'food101': 'food',
    'dtd': 'DTD',
    'aircraft': 'aircraft',
    'ucf101': 'UCF101',
}

tpt_to_iclr_map = {
    'I': 'ImageNet',
    'eurosat': 'EuroSAT',
    'dtd': 'DTD',
    'pets': 'pets',
    'food101': 'food101'
}

tpt_to_gpt4_map = {
    'I': 'ImageNet',
    'flower102': 'flowers',
    'food101': 'food101',
    'dtd': 'dtd',
    'aircraft': 'aircraft',
    'ucf101': 'ucf101',
    'eurosat': 'eurosat-new',
    'caltech101': 'caltech101',
    'cars': 'cars',
    'pets': 'pets',
    'sun397': 'sun397'
}

TPT_TO_CONCEPT_TYPE_MAP = {'labo': tpt_to_labo_map, 'iclr': tpt_to_iclr_map, 'iclr_no_cond': tpt_to_iclr_map, 'regular': tpt_to_regular_map, 'gpt4': tpt_to_gpt4_map, 'gpt4_x_templates': tpt_to_gpt4_map, 'gpt4_no_cond': tpt_to_gpt4_map}

imagenet_vars = ['A', 'R', 'V', 'K']

for concept, map in TPT_TO_CONCEPT_TYPE_MAP.items():
    for i_var in imagenet_vars:
        TPT_TO_CONCEPT_TYPE_MAP[concept][i_var] = map['I']

    new_map = {}
    for k,v in map.items():
        new_map[k + "_sub"] = v
    
    TPT_TO_CONCEPT_TYPE_MAP[concept].update(new_map)

def get_concept_embeds_path(test_set, concept_type):
    embed_dir = CONCEPT_EMBED_DIR_TEMPLATE.format(parent_dir=parent_dir, concept_type=concept_type)
    if test_set not in TPT_TO_CONCEPT_TYPE_MAP[concept_type]:
        raise ValueError(f"dataset {test_set} doesn't have embeddings of type {concept_type}")
    concept_test_set = TPT_TO_CONCEPT_TYPE_MAP[concept_type][test_set]

    return os.path.join(embed_dir, f"{concept_test_set}.pkl")

def get_class_embeds_path(test_set, with_templates=False, with_coop=False):
    concept_test_set = TPT_TO_CONCEPT_TYPE_MAP['regular'][test_set]

    if with_templates:
        dir = CLASS_EMBED_W_TEMPLATES_DIR
    elif with_coop:
        dir = COOP_EMBED_DIR
    else:
        dir = CLASS_EMBED_DIR
    
    return os.path.join(dir, f"{concept_test_set}.pkl")

def get_class2concept_dict_path(test_set, concept_type):
    dict_dir = CLASS2CONCEPT_DICT_TEMPLATE.format(parent_dir=parent_dir, concept_type=concept_type)

    if test_set not in TPT_TO_CONCEPT_TYPE_MAP[concept_type]:
        raise ValueError(f"dataset {test_set} doesn't have embeddings of type {concept_type}")

    concept_test_set = TPT_TO_CONCEPT_TYPE_MAP[concept_type][test_set]

    return os.path.join(dict_dir, f"{concept_test_set}.json")

def get_proj_matrix_path(test_set):
    return os.path.join(PROJ_MATRIX_DIR, f'{test_set}.pkl')