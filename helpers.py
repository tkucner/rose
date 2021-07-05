import json
import sys

from jsonschema import Draft7Validator, validators, exceptions


def extended_validator(json_path, schema_path):
    schema_file = open(schema_path, 'r')
    my_schema = json.load(schema_file)

    json_file = open(json_path, 'r')
    my_json = json.load(json_file)

    validate_properties = Draft7Validator.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for prop, sub_schema in properties.items():
            if "default" in sub_schema:
                instance.setdefault(prop, sub_schema["default"])

        for error in validate_properties(
                validator, properties, instance, schema,
        ):
            yield error

    ext_validator = validators.extend(Draft7Validator, {"properties": set_defaults}, )

    try:
        ext_validator(my_schema).validate(my_json)
    except exceptions.ValidationError as e:
        return False, e
    except exceptions.SchemaError as e:
        return False, e
    return True, my_json


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)