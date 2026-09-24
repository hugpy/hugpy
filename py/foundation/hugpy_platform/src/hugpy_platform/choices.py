"""Small value selection helpers."""


def first_not_none(explicit, from_preset, schema_default):
    if explicit is not None:
        return explicit
    if from_preset is not None:
        return from_preset
    return schema_default
