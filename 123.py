import os
import json
import argparse
import subprocess
from typing import List, Dict, Tuple
import re

class FileSection:
    def __init__(self, section_header: str):
        self.section_header = section_header
        self.dict = {}

    def entered_section(self, line: str):
        s = re.search(self.section_header, line)
        return s is not None

    def parse_line(self, line: str):
        def parse_kv_line(line: str) -> Tuple[Any, Any]:
            """Parse a log line that reports a key-value pair.

            The log line has this format: [mm/dd/yyyy-hh:mm:ss] [I] key_name: key_value
            """
            match = re.search(r'(\[\d+/\d+/\d+-\d+:\d+:\d+\] \[I\] )', line)
            if match is not None:
                match_end = match.span()[1]
                kv_line = line[match_end:].strip()
                kv = kv_line.split(": ")
                if len(kv) > 1:
                    return kv[0], kv[1]
            return None, None

        k,v = parse_kv_line(line)
        if k is not None and v is not None:
            self.dict[k] = v
            return True
        if k is not None:
            return True
        return False

def __parse_log_file(file_name: str, sections: List) -> List[Dict]:
    current_section = None
    with open(file_name, "r") as file:
        for line in file.readlines():
            if current_section is None:
                for section in sections:
                    if section.entered_section(line):
                        current_section = section
                        break
            else:
                if not current_section.parse_line(line):
                    current_section = None
    dicts = [section.dict for section in sections]
    return dicts

def parse_build_log(file_name: str) -> List[Dict]:
    """Parse the TensorRT engine build log and extract the builder configuration.

    Returns the model and engine build configurations as dictionaries.
    """
    model_options = FileSection("=== Model Options ===")
    build_options = FileSection("=== Build Options ===")
    sections = [model_options, build_options]
    __parse_log_file(file_name, sections)
    return {
        "model_options": model_options.dict,
        "build_options": build_options.dict,
    }


def generate_build_metadata(log_file: str, output_json: str):
    """Parse trtexec engine build log file and write to a JSON file"""
    build_metadata = parse_build_log(log_file)
    with open(output_json, 'w') as fout:
        json.dump(build_metadata , fout)
        print(f"Engine building metadata: generated output file {output_json}")