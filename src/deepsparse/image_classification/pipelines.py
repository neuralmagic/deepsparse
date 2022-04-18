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

"""
Image classification pipeline
"""
__status__ = "Under-Development"
try:
    import torch

    torch_error = None
except ModuleNotFoundError as error:
    torch = None
    torch_error = error


def main():
    print(f"Currently this module is {__status__}")
    if torch:
        print("Torch version:", torch.__version__)


if __name__ == '__main__':
    main()