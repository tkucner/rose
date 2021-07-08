import json
import os
import sys
from datetime import datetime

from jsonschema import Draft7Validator, validators, exceptions


def file_path_formatter(directory, file):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    map_file_name_directory = os.path.splitext(os.path.basename(file))[0] + "_" + dt_string
    save_directory = os.path.join(directory, map_file_name_directory)
    if os.path.isdir(save_directory):
        print("Directory exits, data will be over written")
    else:
        os.makedirs(save_directory)
    return save_directory


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