# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

PROMPT_TEMPLATES = {
    'base': {
        "template": "{task_template}",
        "stop_words": ["<|end_of_text|>"],
    },
    'llama3': {
        "template": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{task_template}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        "stop_words": ["<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"],
    },
    'qwen3': {
        "template": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{task_template}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "stop_words": ["<|im_end|>", "<|endoftext|>"],
    },
}
