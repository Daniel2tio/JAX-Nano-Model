"""Launches the story-generation Gradio UI outside Jupyter.

Jupyter applies nest_asyncio, which breaks anyio/Starlette on Python 3.12+ and
causes HTTP 500 errors when Gradio is launched in-notebook. Run this script
from a normal process instead:

    uv run python gradio_demo.py
"""

from pathlib import Path

import gradio as gr
import jax
import flax.nnx as nnx
from jax.sharding import SingleDeviceSharding
from orbax import checkpoint

from helper import NanoJAX, generate_story


def load_model(checkpoint_path: Path) -> NanoJAX:
    model = NanoJAX()
    cpu_sharding = SingleDeviceSharding(jax.devices("cpu")[0])
    restore_args = jax.tree_util.tree_map(
        lambda _: checkpoint.ArrayRestoreArgs(sharding=cpu_sharding),
        nnx.state(model),
    )
    checkpointer = checkpoint.PyTreeCheckpointer()
    restored_state = checkpointer.restore(
        checkpoint_path,
        item=nnx.state(model),
        restore_args=restore_args,
    )
    nnx.update(model, restored_state)
    return model


def main() -> None:
    checkpoint_path = Path(__file__).resolve().parent / "small_checkpoint.orbax"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    model = load_model(checkpoint_path)

    def create_story(story_prompt: str, temperature: float, max_new_tokens: int) -> str:
        return generate_story(model, story_prompt, temperature, max_new_tokens)

    demo = gr.Interface(
        fn=create_story,
        inputs=[
            gr.Textbox(label="Story Prompt"),
            gr.Slider(minimum=0, maximum=1.0, value=0.8, step=0.01, label="Temperature"),
            gr.Slider(minimum=0, maximum=200, value=10, step=1, label="Max Tokens"),
        ],
        outputs="text",
    )
    demo.launch(share=False)


if __name__ == "__main__":
    main()
