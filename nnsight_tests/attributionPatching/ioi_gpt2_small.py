import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import einops
import torch
import plotly.express as px
from profiler import Profiler
from nnsight import LanguageModel

profiler = Profiler()


prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]

answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]

@profiler.log_profile
def get_logit_diff(logits, answer_token_indices):
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()

@profiler.log_profile
def trace_model(tokens, num_tokens=None, num_prompts=None):
    if num_tokens is None:
        num_tokens = tokens.shape[1]
    if num_prompts is None:
        num_prompts = tokens.shape[0]
    return model.trace(tokens, trace=False).logits.cpu(), num_tokens, num_prompts

@profiler.log_profile
def compute_baselines():
    clean_logits, num_tokens, num_prompts = trace_model(clean_tokens, num_tokens=clean_tokens.shape[1], num_prompts=clean_tokens.shape[0])
    corrupted_logits, _, _ = trace_model(corrupted_tokens, num_tokens=corrupted_tokens.shape[1], num_prompts=corrupted_tokens.shape[0])

    clean_baseline = get_logit_diff(clean_logits, answer_token_indices).item()
    print(f"Clean logit diff: {clean_baseline:.4f}")

    corrupted_baseline = get_logit_diff(corrupted_logits, answer_token_indices).item()
    print(f"Corrupted logit diff: {corrupted_baseline:.4f}")
    
    return clean_baseline, corrupted_baseline, clean_logits, corrupted_logits, num_tokens, num_prompts

@profiler.log_profile
def trace_attention():
    clean_out = []
    corrupted_out = []
    corrupted_grads = []

    with model.trace() as tracer:

        with tracer.invoke(clean_tokens) as invoker_clean:
            for layer in model.transformer.h:
                attn_out = layer.attn.c_proj.input[0][0]
                clean_out.append(attn_out.save())

        with tracer.invoke(corrupted_tokens) as invoker_corrupted:
            for layer in model.transformer.h:
                attn_out = layer.attn.c_proj.input[0][0]
                corrupted_out.append(attn_out.save())
                corrupted_grads.append(attn_out.grad.save())

            logits = model.lm_head.output.save()
            value = ioi_metric(logits.cpu())
            value.backward()

    return clean_out, corrupted_out, corrupted_grads

@profiler.log_profile
def patch_attention_heads():
    patching_results = []

    for corrupted_grad, corrupted, clean, layer in zip(
        corrupted_grads, corrupted_out, clean_out, range(len(clean_out))
    ):
        residual_attr = einops.reduce(
            corrupted_grad.value[:,-1,:] * (clean.value[:,-1,:] - corrupted.value[:,-1,:]),
            "batch (head dim) -> head",
            "sum",
            head=12,
            dim=64,
        )
        patching_results.append(residual_attr.detach().cpu().numpy())

    fig = px.imshow(
        patching_results,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        title="Patching Over Attention Heads"
    )
    fig.update_layout(
        xaxis_title="Head",
        yaxis_title="Layer"
    )
    fig.show()



@profiler.log_profile
def patch_positions():
    patching_results = []

    for corrupted_grad, corrupted, clean, layer in zip(
        corrupted_grads, corrupted_out, clean_out, range(len(clean_out))
    ):
        residual_attr = einops.reduce(
            corrupted_grad.value * (clean.value - corrupted.value),
            "batch pos dim -> pos",
            "sum",
        )
        patching_results.append(residual_attr.detach().cpu().numpy())

    fig = px.imshow(
        patching_results,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
        title="Patching Over Position"
    )
    fig.update_layout(
        xaxis_title="Position",
        yaxis_title="Layer"
    )
    fig.show()




if __name__ == "__main__":
    

    # Print available devices
    devices = profiler.available_devices()
    for device_name, device_str in devices.items():
        print(f"Device: {device_name}, String: {device_str}")

    model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

    clean_tokens = model.tokenizer(prompts, return_tensors="pt")["input_ids"]

    corrupted_tokens = clean_tokens[
        [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
    ]

    answer_token_indices = torch.tensor(
        [
            [model.tokenizer(answers[i][j])["input_ids"][0] for j in range(2)]
            for i in range(len(answers))
        ]
    )
    # Check device of the model
    print(model.device)

    clean_baseline, corrupted_baseline, clean_logits, corrupted_logits, num_tokens, num_prompts = compute_baselines()

    @profiler.log_profile
    def ioi_metric(logits, answer_token_indices=answer_token_indices):
        return (get_logit_diff(logits, answer_token_indices) - corrupted_baseline) / (
            clean_baseline - corrupted_baseline
        )

    print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
    print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")

    clean_out, corrupted_out, corrupted_grads = trace_attention()

    patch_attention_heads()

    patch_positions()