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
from typing import Iterator, Optional


class InvalidPromptSourceDirectoryException(Exception):
    pass


class PromptParser:
    class Extentions(Enum):
        TEXT = ".txt"
        CSV = ".csv"
        JSON = ".json"
        JSONL = ".jsonl"

    def __init__(self, filename: str):
        self.extention = self._validate_and_return_extention(filename)
        self.filename: str = filename

    def parse_as_iterable(self):

        if self.extention == self.Extentions.TEXT:
            return self._parse_text()
        if self.extention == self.Extentions.CSV:
            return self._parse_csv()
        if self.extention == self.Extentions.JSON:
            return self._parse_json_list()
        if self.extention == self.Extentions.JSONL:
            return self._parse_jsonl()

    def _parse_text(self):
        with open(self.filename, "r") as file:
            for line in file:
                yield line.strip()

    def _parse_csv(self, column_name: str = "prompt"):
        with open(self.filename, "r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                yield row

    def _parse_json_list(self):
        with open(self.filename, "r") as file:
            json_list = json.load(file)
            for json_object in json_list:
                yield json_object

    def _parse_jsonl(self):
        with open(self.filename, "r") as file:
            for jsonl in file:
                yield json.loads(jsonl)

    def _validate_and_return_extention(self, filename: str):
        if os.path.exists(filename):

            for extention in self.Extentions:
                if filename.endswith(extention.value):
                    return extention

            raise InvalidPromptSourceDirectoryException(
                f"{filename} is not a valid source extract batched prompts"
            )
        raise FileNotFoundError
