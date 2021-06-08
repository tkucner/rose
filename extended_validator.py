import json

from jsonschema import Draft7Validator, validators, exceptions


class ExtendedValidator:
    def __init__(self, jason_path, schema_path):
        """
        Args:
            jason_path:
            schema_path:
        """
        self.my_schema = self.__load_json(schema_path)
        self.my_json = self.__load_json(jason_path)

    def extended_validator(self):
        DefaultValidatingDraft7Validator = self.__extend_with_default(Draft7Validator)

        try:
            DefaultValidatingDraft7Validator(self.my_schema).validate(self.my_json)
        except exceptions.ValidationError as e:
            return False, e
        except exceptions.SchemaError as e:
            return False, e
        return True, self.my_json

    @staticmethod
    def __extend_with_default(validator_class):
        """
        Args:
            validator_class:
        """
        validate_properties = validator_class.VALIDATORS["properties"]

        def set_defaults(validator, properties, instance, schema):
            for property, subschema in properties.items():
                if "default" in subschema:
                    instance.setdefault(property, subschema["default"])

            for error in validate_properties(
                    validator, properties, instance, schema,
            ):
                yield error

        return validators.extend(
            validator_class, {"properties": set_defaults},
        )

    @staticmethod
    def __load_json(path):
        """
        Args:
            path:
        """
        with open(path, 'r') as f:
            my_json_file = json.load(f)
            return my_json_file
