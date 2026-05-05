"""
PhysEdit Python API Example

Demonstrates programmatic usage of PhysEdit for image editing.
"""

import sys
sys.path.insert(0, "..")

from api import PhysEditAPI


def main():
    # Initialize API
    api = PhysEditAPI(
        model_path="checkpoints/ChronoEdit-14B-Diffusers",
        offload=True,  # Use CPU offloading if VRAM < 40GB
    )

    # --- Example 1: Auto-detect complexity ---
    print("Example 1: Auto complexity detection")
    result = api.edit(
        image="examples/input_color.jpg",
        prompt="Change the hat to a red cap",
    )
    result.save("outputs/api_auto.jpg")
    print("  Saved to outputs/api_auto.jpg")

    # --- Example 2: Manual high-complexity ---
    print("Example 2: Manual high complexity")
    result = api.edit(
        image="examples/input_robot.jpg",
        prompt="The robot picks up the cup",
        card_override_nr=15,
        card_override_r=8,
    )
    result.save("outputs/api_high.jpg")
    print("  Saved to outputs/api_high.jpg")

    # --- Example 3: Tune SRM for precise local edit ---
    print("Example 3: Precise local edit with sharp SRM")
    result = api.edit(
        image="examples/input_portrait.jpg",
        prompt="Change eye color to green",
        srm_temperature=0.05,  # Sharp mask for very local edit
    )
    result.save("outputs/api_local.jpg")
    print("  Saved to outputs/api_local.jpg")

    # --- Example 4: Batch editing ---
    print("Example 4: Batch editing")
    edits = [
        ("input.jpg", "Add sunglasses"),
        ("input.jpg", "Change to winter scene"),
        ("input.jpg", "Superhero pose"),
    ]
    for i, (img, prompt) in enumerate(edits):
        result = api.edit(image=f"examples/{img}", prompt=prompt, seed=42+i)
        result.save(f"outputs/batch_{i}.jpg")
        print(f"  [{i+1}/{len(edits)}] '{prompt}' -> outputs/batch_{i}.jpg")


if __name__ == "__main__":
    main()
