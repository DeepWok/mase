from sys import version_info
from typing import Any, List, Optional, Type
from dataclasses import Field, fields, is_dataclass
from enum import Enum
from textwrap import indent
ALL_SKIP_FIELD = "is_perfect"
FIELD_MAP = {"forward": "forward_io", "backward": "backward_io"}
ALWAYS_INCLUDE = ["forward", "backward", "update"]

class _PrintableMixin:
    """Helper class for pretty-printing of config dataclasses."""

    # pylint: disable=too-few-public-methods

    def __str__(self) -> str:
        """Return a pretty-print representation."""

        def lines_list_to_str(
            lines_list: List[str],
            prefix: str = "",
            suffix: str = "",
            indent_: int = 0,
            force_multiline: bool = False,
        ) -> str:
            """Convert a list of lines into a string.

            Args:
                lines_list: the list of lines to be converted.
                prefix: an optional prefix to be appended at the beginning of
                    the string.
                suffix: an optional suffix to be appended at the end of the string.
                indent_: the optional number of spaces to indent the code.
                force_multiline: force the output to be multiline.

            Returns:
                The lines collapsed into a single string (potentially with line
                breaks).
            """
            if force_multiline or len(lines_list) > 3 or any("\n" in line for line in lines_list):
                # Return a multi-line string.
                lines_str = indent(",\n".join(lines_list), " " * indent_)
                prefix = "{}\n".format(prefix) if prefix else prefix
                suffix = "\n{}".format(suffix) if suffix else suffix
            else:
                # Return an inline string.
                lines_str = ", ".join(lines_list)

            return "{}{}{}".format(prefix, lines_str, suffix)

        def field_to_str(field_value: Any) -> str:
            """Return a string representation of the value of a field.

            Args:
                field_value: the object that contains a field value.

            Returns:
                The string representation of the field (potentially with line
                breaks).
            """
            field_lines = []
            force_multiline = False

            # Handle special cases.
            if isinstance(field_value, list) and len(value) > 0:
                # For non-empty lists, always use multiline, with one item per line.
                for item in field_value:
                    field_lines.append(indent("{}".format(str(item)), " " * 4))
                force_multiline = True
            else:
                field_lines.append(str(field_value))

            prefix = "[" if force_multiline else ""
            suffix = "]" if force_multiline else ""
            return lines_list_to_str(field_lines, prefix, suffix, force_multiline=force_multiline)

        def is_skippable(field: Field, value: Any) -> bool:
            """Return whether a field should be skipped."""
            if field.metadata.get("always_show", False):
                return False

            if value == field.default:
                # Skip fields with the default value.
                return True

            if "hide_if" in field.metadata and field.metadata.get("hide_if") == value:
                return True

            return False

        # Main loop.

        # Build the list of lines.
        fields_lines = []

        # special case for global skip:
        all_skip = hasattr(self, ALL_SKIP_FIELD) and getattr(self, ALL_SKIP_FIELD)

        for field in fields(self):  # type: ignore[arg-type]
            value = getattr(self, field.name)

            # Exclude fields.
            if (all_skip and field.name != ALL_SKIP_FIELD) or is_skippable(field, value):
                continue

            # Convert the value into a string, falling back to repr if needed.
            try:
                value_str = field_to_str(value)
            except Exception:  # pylint: disable=broad-except
                value_str = str(value)

            fields_lines.append("{}={}".format(field.name, value_str))

        # Convert the full object to str.
        output = lines_list_to_str(fields_lines, "{}(".format(self.__class__.__name__), ")", 4)

        return output
