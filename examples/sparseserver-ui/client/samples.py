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

sample = {
    "example 1": {
        "context": (
            "The DeepSparse Engine is a CPU runtime that delivers"
            "GPU-class performance by taking advantage of sparsity within neural"
            "networks to reduce compute required as well as accelerate memory bound"
            "workloads. It is focused on model deployment and scaling machine"
            "learning pipelines, fitting seamlessly into your existing deployments"
            "as an inference backend."
        ),
        "question": (
            "What does the DeepSparse Engine take advantage of within neural networks?"
        ),
    },
    "example 2": {
        "context": (
            "Concerns were raised over whether Levi's Stadium's field was of a high"
            "enough quality to host a Super Bowl; during the inaugural season, the field"
            "had to be re-sodded multiple times due to various issues, and during a week"
            "6 game earlier in the 2015 season, a portion of the turf collapsed under"
            "Baltimore Ravens kicker Justin Tucker, causing him to slip and miss a field"
            "goal."
        ),
        "question": ("What collapsed on Justin Tucker?"),
    },
    "example 3": {
        "context": (
            "The league announced on October 16, 2012, that the two finalists were Sun"
            "Life Stadium and Levi's Stadium. The South Florida/Miami area has previously"
            "hosted the event 10 times (tied for most with New Orleans), with the"
            "most recent one being Super Bowl XLIV in 2010. The San Francisco Bay"
            "Area last hosted in 1985 (Super Bowl XIX), held at Stanford Stadium in"
            "Stanford, California, won by the home team 49ers. The Miami bid depended"
            "on whether the stadium underwent renovations."
        ),
        "question": (
            "What was the most recent Super Bowl that took place at Sun "
            "Life Stadium in Miami?"
        ),
    },
}
