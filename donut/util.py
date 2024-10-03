"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import torch
import zss
from datasets import load_dataset
from nltk import edit_distance
from torch.utils.data import Dataset
from transformers.modeling_utils import PreTrainedModel
from zss import Node


def save_json(write_path: Union[str, bytes, os.PathLike], save_obj: Any):
    with open(write_path, "w") as f:
        json.dump(save_obj, f)

def load_json(json_path: Union[str, bytes, os.PathLike]):
    with open(json_path, "r") as f:
        return json.load(f)

# Add a debug print to track custom_collate_fn
'''
def custom_collate_fn(batch):
    print("DEBUG: Entered custom_collate_fn")
    input_tensors, input_id_chunks, labels_chunks = zip(*batch)
    print("DEBUG: Batch unpacked in custom_collate_fn")
    print("DEBUG: Printing input_tensors", input_tensors)
    print("DEBUG: Printing input_id_chunks", input_id_chunks)
    print("DEBUG: Printing labels_chunks", labels_chunks)

    max_length_chunk = max(len(chunks) for chunks in input_id_chunks)
    input_tensors = input_tensor 
    image_size = input_tensors.view(1) * input_tensors.view(2)
    max_attempts = 100s[0]
   
    input_tensors_padded = pad_sequence(input_tensors, batch_first=True, padding_value=0)
    print("DEBUG: Padded input tensors")

    height, width = input_tensors_padded.size(1), input_tensors_padded.size(2)
    attempts = 0  # Added initialization for attempts
    while (height * width) % (64 * 16 * max_length_chunk) != 0:
        max_length_chunk += 1
        attempts += 1
        if attempts >= max_attempts:
            print(f"Condition not met after {max_attempts} attempts. Current max_length_chunk: {max_length_chunk}")
            break
    print("DEBUG: max_length_chunk updated to:", max_length_chunk)

    input_ids_padded = []
    for chunk_list in input_id_chunks:
        padded_chunks = pad_sequence(chunk_list, batch_first=True, padding_value=0)
        if len(chunk_list) < max_length_chunk:
            padding_chunks = torch.zeros((max_length_chunk - len(chunk_list), padded_chunks.size(1)), dtype=torch.long)
            padded_chunks = torch.cat([padded_chunks, padding_chunks], dim=0)
        input_ids_padded.append(padded_chunks)
    input_ids_padded = torch.stack(input_ids_padded)
    print("DEBUG: input_ids_padded created")

    labels_padded = []
    for chunk_list in labels_chunks:
        padded_chunks = pad_sequence(chunk_list, batch_first=True, padding_value=-100)  # Use ignore_id for padding
        if len(chunk_list) < max_length_chunk:
            padding_chunks = torch.full((max_length_chunk - len(chunk_list), padded_chunks.size(1)), -100, dtype=torch.long)
            padded_chunks = torch.cat([padded_chunks, padding_chunks], dim=0)
        labels_padded.append(padded_chunks)
    labels_padded = torch.stack(labels_padded)
    print("DEBUG: labels_padded created")

    return input_tensors_padded, input_ids_padded, labels_padded
'''
class DonutDataset(Dataset):
    def __init__(
        self,
        dataset_name_or_path: str,
        donut_model: PreTrainedModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
    ):
        super().__init__()
        self.donut_model = donut_model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key

        print("DEBUG: Loading dataset:", dataset_name_or_path)
        self.dataset = load_dataset(dataset_name_or_path, split=self.split)  # This line may be a bottleneck
        self.dataset_length = len(self.dataset)
        print("DEBUG: Loaded dataset with length:", self.dataset_length)

        self.gt_token_sequences = []

        # Main loop to process the dataset
        loop = 0
        for sample in self.dataset:
            loop += 1
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            # Generate token sequences without dynamically adding special tokens
            self.gt_token_sequences.append(
                [
                    task_start_token
                    + self.donut_model.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=False,  # Disable dynamic addition
                        sort_json_key=self.sort_json_key,
                    )
                    + self.donut_model.decoder.tokenizer.eos_token
                    for gt_json in gt_jsons
                ]
            )

            # Print checkpoint every 500 iterations for progress monitoring
            if loop % 500 == 0:
                print(f'DEBUG: Processed {loop} samples')

        # Add task-specific tokens if they haven't been added already
        self.donut_model.decoder.add_special_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        print("DEBUG: Finished processing the dataset")

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        print("DEBUG: __getitem__ called with index:", idx)
        sample = self.dataset[idx]

        # input_tensor
        input_tensor = self.donut_model.encoder.prepare_input(sample["image"], random_padding=self.split == "train")
        processed_parse = random.choice(self.gt_token_sequences[idx])
        input_ids = self.donut_model.decoder.tokenizer(
            processed_parse,
            add_special_tokens=False,
            padding = 'max_length',
            max_length = self.max_length,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        print("DEBUG: Input IDs created")
        '''
        # Chunk input_ids into smaller pieces
        input_chunks = self._chunk_input_ids(input_ids)
        if self.split == "train":
            labels_chunks = []
            for chunk in input_chunks:
                labels = chunk.clone()
                labels[labels == self.donut_model.decoder.tokenizer.pad_token_id] = self.ignore_id
                labels[: torch.nonzero(labels == self.prompt_end_token_id).sum() + 1] = self.ignore_id
                labels_chunks.append(labels)
            print("DEBUG: Returning train data for index:", idx)
            return input_tensor, input_chunks, labels_chunks
        else:
            prompt_end_index = torch.nonzero(input_chunks[0] == self.prompt_end_token_id).sum()
            print("DEBUG: Returning validation data for index:", idx)
            return input_tensor, input_chunks, prompt_end_index, processed_parse
        '''
        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.donut_model.decoder.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse



    def _chunk_input_ids(self, input_ids):
        """Chunk input_ids into smaller sequences based on max_length."""
        print("DEBUG: Entering _chunk_input_ids")
        input_chunks = []
        for i in range(0, len(input_ids), self.max_length):
            chunk = input_ids[i: i + self.max_length]
            if len(chunk) < self.max_length:  # Add padding if the chunk is smaller than max_length
                padding_length = self.max_length - len(chunk)
                chunk = torch.cat([chunk, torch.full((padding_length,), self.donut_model.decoder.tokenizer.pad_token_id)])
            input_chunks.append(chunk)
        print(f"DEBUG: Created {len(input_chunks)} input chunks")
        return input_chunks

