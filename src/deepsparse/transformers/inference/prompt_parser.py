# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import csv
import json
import os
from enum import Enum


class InvalidPromptSourceDirectoryException(Exception):
    pass


class UnableToParseExtentionException(Exception):
    pass


class PromptParser:
    class Extensions(Enum):
        TEXT = ".txt"
        CSV = ".csv"
        JSON = ".json"
        JSONL = ".jsonl"

    def __init__(self, filename: str):
        self.extention: self.Extensions = self._validate_and_return_extention(filename)
        self.filename: str = filename

    def parse_as_iterable(self):

        if self.extention == self.Extensions.TEXT:
            return self._parse_text()
        if self.extention == self.Extensions.CSV:
            return self._parse_csv()
        if self.extention == self.Extensions.JSON:
            return self._parse_json_list()
        if self.extention == self.Extensions.JSONL:
            return self._parse_jsonl()

        raise UnableToParseExtentionException(
            f"Parser for {self.extention} does not exist"
        )

    def _parse_text(self):
        with open(self.filename, "r") as file:
            for line in file:
                yield line.strip(), {}

    def _parse_csv(self):
        with open(self.filename, "r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                yield row.get("prompt"), row

    def _parse_json_list(self):
        with open(self.filename, "r") as file:
            json_list = json.load(file)
            for json_object in json_list:
                yield json_object.get("prompt"), json_object

    def _parse_jsonl(self):
        with open(self.filename, "r") as file:
            for jsonl in file:
                jsonl_object = json.loads(jsonl)
                yield jsonl_object.get("prompt"), jsonl_object

    def _validate_and_return_extention(self, filename: str):
        if os.path.exists(filename):

            for extention in self.Extensions:
                if filename.endswith(extention.value):
                    return extention

            raise InvalidPromptSourceDirectoryException(
                f"{filename} is not compatible. Select file that has "
                "extension from "
                f"{[key.name for key in self.Extensions]}"
            )
        raise FileNotFoundError
