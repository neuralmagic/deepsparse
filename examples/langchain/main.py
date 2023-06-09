from typing import Any, List, Mapping, Optional
import requests
import json

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain import PromptTemplate, LLMChain
from langchain.chains import SimpleSequentialChain

class DeepSparseLLM(LLM):
    url: str
        
    @property
    def _llm_type(self) -> str:
        return "deepsparse"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        obj = {"sequences": prompt}

        response = requests.post(self.url, json=obj)
        json_response = json.loads(response.text)
        output = json.loads(response.text)["labels"][0]

        return output
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"url": self.url}

prompt_template = "Today I am feeling {adjective}."
url = "http://localhost:5543/predict" # Server's port default to 5543

llm = DeepSparseLLM(url=url)
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

input_list = [
    {"adjective": "good"},
]

overall_chain = SimpleSequentialChain(chains=[llm_chain], verbose=True)
print(overall_chain.run(input_list))

"""
input_list = [
    {"adjective": "good"},
    {"adjective": "sad"},
    {"adjective": "great"}
]

print(llm_chain.apply(input_list))
"""

