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
from typing import List, Type

from pydantic import BaseModel, Field
from transformers import (
    BeamSearchScorer,
    LogitsProcessorList,
    MaxLengthCriteria,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    StoppingCriteriaList,
)

import open_clip
import torch
import torch.nn.functional as F
from deepsparse.clip import CLIPDecoderInput, CLIPTextInput, CLIPVisualInput
from deepsparse.pipeline import BasePipeline, Pipeline


__all__ = ["CLIPCaptionInput", "CLIPCaptionOutput", "CLIPCaptionPipeline"]


class CLIPCaptionInput(BaseModel):
    """
    Input for the CLIP Caption Pipeline
    """

    image: CLIPVisualInput = Field(
        description="Path to image to run zero-shot prediction on."
    )


class CLIPCaptionOutput(BaseModel):
    """
    Output for the CLIP Caption Output
    """

    caption: List[str] = Field(description="Caption produced for the given image.")


@BasePipeline.register(task="clip_caption", default_model_path=None)
class CLIPCaptionPipeline(BasePipeline):
    """
    Pipelines designed to generate a caption for a given image. The CLIPCaptionPipeline
    relies on 3 other pipelines: CLIPVisualPipeline, CLIPTextPipeline, and the
    CLIPDecoder Pipeline. The pipeline takes in a single image and then uses the
    pipelines along with Beam Search to generate a caption.

    :param visual_model_path: either a local path or sparsezoo stub for the CLIP visual
    branch onnx model
    :param text_model_path: either a local path or sparsezoo stub for the CLIP text
    branch onnx model
    :param decoder_model_path: either a local path or sparsezoo stub for the CLIP
    decoder branch onnx model
    :param num_beams: number of beams to use in Beam Search
    :param num_beam_groups: number of beam groups to use in Beam Search
    :param min_seq_len: the minimum length of the caption sequence
    :param max_seq_len: the maxmium length of the caption sequence
    :param pipeline_engine_args: dictionary of arguments specific to running the
    pipeline engine. Applied to the text branch, visual branch, and decoder.
    See pipeline.py for a full list of arguments.

    """

    def __init__(
        self,
        visual_model_path: str,
        text_model_path: str,
        decoder_model_path: str,
        num_beams: int = 10,
        num_beam_groups: int = 5,
        min_seq_len: int = 5,
        max_seq_len: int = 20,
        pipeline_engine_args: dict = None,
        **kwargs,
    ):
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

        if pipeline_engine_args is None:
            pipeline_engine_args = {}

        pipeline_engine_args["model_path"] = visual_model_path
        self.visual = Pipeline.create(
            task="clip_visual",
            **pipeline_engine_args,
        )
        pipeline_engine_args["model_path"] = text_model_path
        self.text = Pipeline.create(
            task="clip_text",
            **pipeline_engine_args,
        )
        pipeline_engine_args["model_path"] = decoder_model_path
        self.decoder = Pipeline.create(
            task="clip_decoder",
            **pipeline_engine_args,
        )

        super().__init__(**kwargs)

    def _encode_and_decode(self, text: torch.Tensor, image_embs: torch.Tensor):
        original_size = text.shape[-1]
        padded_tokens = F.pad(text, (15 - original_size, 0))
        text_embeddings = self.text(
            CLIPTextInput(text=padded_tokens.numpy())
        ).text_embeddings
        _, text_embs = text_embeddings[0], text_embeddings[1]
        logits = self.decoder(
            CLIPDecoderInput(
                image_embeddings=image_embs.numpy(), text_embeddings=text_embs
            )
        ).logits
        return {
            "logits": torch.Tensor(logits[0]),
        }

    # Adapted from open_clip
    def _generate(self, pipeline_inputs: CLIPCaptionInput):
        sot_token_id = 49406
        eos_token_id = 49407
        pad_token_id = 0
        batch_size = 1
        repetition_penalty = 1.0
        device = "cpu"

        stopping_criteria = [MaxLengthCriteria(max_length=self.max_seq_len)]
        stopping_criteria = StoppingCriteriaList(stopping_criteria)

        logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(self.min_seq_len, eos_token_id),
                RepetitionPenaltyLogitsProcessor(repetition_penalty),
            ]
        )

        visual_output = self.visual(pipeline_inputs.image).image_embeddings

        _, image_embs = visual_output[0], visual_output[1]

        image_embs = torch.repeat_interleave(
            torch.Tensor(image_embs), self.num_beams, dim=0
        )

        input_ids = torch.ones(
            (batch_size * self.num_beams, 1), device=device, dtype=torch.long
        )
        input_ids = input_ids * sot_token_id
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=self.num_beams,
            device=device,
            num_beam_groups=self.num_beam_groups,
        )

        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_beam_size, cur_len = input_ids.shape
        beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, "
                f"but is {batch_beam_size}."
            )

        beam_scores = torch.full(
            (batch_size, num_beams), -1e9, dtype=torch.float, device=device
        )
        # initialise score of first beam of each group with 0 and the rest with 1e-9.
        # This ensures that the beams in the same group don't produce same tokens
        # everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:

            # predicted tokens in cur_len step
            current_tokens = torch.zeros(
                batch_size * num_beams, dtype=input_ids.dtype, device=device
            )

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(
                batch_size * num_beams, dtype=torch.long, device=device
            )

            # do one decoder step on all beams of all sentences in batch
            outputs = self._encode_and_decode(text=input_ids, image_embs=image_embs)

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [
                            batch_idx * num_beams + idx
                            for idx in range(group_start_idx, group_end_idx)
                        ]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of currentg group only
                next_token_logits = outputs["logits"][batch_group_indices, -1, :]
                vocab_size = next_token_logits.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids,
                    next_token_logits,
                    current_tokens=current_tokens,
                    beam_group_idx=beam_group_idx,
                )
                next_token_scores = next_token_scores_processed + beam_scores[
                    batch_group_indices
                ].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(
                    next_token_scores_processed
                )

                # reshape for beam search
                next_token_scores = next_token_scores.view(
                    batch_size, group_size * vocab_size
                )

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = (
                    sum(beam_indices, ()) if beam_indices is not None else None
                )
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat(
                    [group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)],
                    dim=-1,
                )
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch.div(beam_idx, group_size, rounding_mode="floor")
                    + group_start_idx
                    + (beam_idx % group_size)
                )

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            # increase cur_len
            cur_len = cur_len + 1
            if beam_scorer.is_done or stopping_criteria(input_ids, None):
                break

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
        )
        return sequence_outputs["sequences"]

    def __call__(self, *args, **kwargs):
        pipeline_inputs = self.parse_inputs(*args, **kwargs)

        if not isinstance(pipeline_inputs, self.input_schema):
            raise RuntimeError(
                f"Unable to parse {self.__class__} inputs into a "
                f"{self.input_schema} object. Inputs parsed to {type(pipeline_inputs)}"
            )

        output = self._generate(pipeline_inputs)
        output = (
            open_clip.decode(output[0])
            .split("<end_of_text>")[0]
            .replace("<start_of_text>", "")
        )
        return self.output_schema(caption=[output])

    @property
    def input_schema(self) -> Type[CLIPCaptionInput]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return CLIPCaptionInput

    @property
    def output_schema(self) -> Type[CLIPCaptionOutput]:
        """
        :return: pydantic model class that outputs to this pipeline must comply to
        """
        return CLIPCaptionOutput
