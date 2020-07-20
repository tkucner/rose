import argparse
import json

import fastjsonschema


def validateJSON(jsonData):
    try:
        json.loads(jsonData)
    except ValueError as err:
        return False
    return True


def load_file(path):
    with open(path, 'r') as f:
        my_json_file = json.load(f)
        return my_json_file


def validateJSON_schema(json_file, schema_file):
    validator = fastjsonschema.compile(schema_file)
    try:
        validator(json_file)
    except fastjsonschema.JsonSchemaException as e:
        msg = e.message + "\n\nresponse: %s\n\nschema: %s" % (json_file, schema_file)
        return False, msg
    else:
        return True, json_file


parser = argparse.ArgumentParser()
parser.add_argument('schema', help='Path to the schema file')
parser.add_argument('json', help='Path to the json file')
args = parser.parse_args()

my_schema = load_file(args.schema)
my_json = load_file(args.json)
print(my_schema)
print(my_json)

_, msg = validateJSON_schema(my_json, my_schema)
print(msg)

exit(1)
