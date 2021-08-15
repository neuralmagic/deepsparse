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

import ast
import re
from itertools import cycle

import pandas as pd
import streamlit as st
from classification import CORES_PER_SOCKET, Predictor


st.title("DeepSparse Example Application")

model_name = st.sidebar.selectbox(
    "Which model do you like best?",
    Predictor.model_registry.keys(),
)

batch_size = st.sidebar.select_slider(
    "Enter batch size",
    options=[2 ** i for i in range(11)],
    value=64,
)

num_cores = st.sidebar.select_slider(
    "Enter number of cores",
    options=range(1, CORES_PER_SOCKET + 1),
    value=CORES_PER_SOCKET,
)

show_engine_info = st.sidebar.checkbox("Show Engine info")
show_sample_data = st.sidebar.checkbox("Show Sample Data")

st.subheader("Code to run Benchmark")

code = f"""
from classification import Predictor
model_name, batch_size, num_cores = '{model_name}', {batch_size}, {num_cores}
predictor = Predictor(
                    model_name='{model_name}',
                    batch_size={batch_size},
                    num_cores={num_cores},
                    )
results = predictor.benchmark_on_sample_data()
"""

st.code(code, language="python")

if st.button("Execute"):
    subheading = st.text("Fetching Sample Data...")

    predictor = Predictor(
        model_name=model_name, batch_size=batch_size, num_cores=num_cores
    )
    inputs, _, _ = predictor.sample_batch()

    subheading.text("Running Benchmark on Sample Data...")

    results = predictor.benchmark_on_sample_data()
    subheading.text("Benchmark Results...")
    benchmark_results = ast.literal_eval(re.findall("\\(.*?\\)", repr(results))[0])
    st.table(
        pd.DataFrame.from_dict(
            benchmark_results, orient="index", columns=["time in ms"]
        )
    )

    if show_engine_info:
        st.write("Engine Info:")
        st.write(predictor._engine._properties_dict())

    if show_sample_data:
        st.write("Sample Input:")
        cols = st.columns(4)

        for image, col in zip(inputs[0], cycle(cols)):
            with col:
                st.image(
                    image.reshape(224, 224, 3),
                    clamp=True,
                    use_column_width=True,
                )
