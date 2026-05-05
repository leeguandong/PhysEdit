#!/bin/bash
# PhysEdit Example Usage

MODEL_PATH="checkpoints/ChronoEdit-14B-Diffusers"

echo "=== Example 1: Simple color edit (CARD: low complexity, Nr=3, r=2) ==="
python physedit.py \
    -i examples/input_color.jpg \
    -p "Change the hat to a bright red cap" \
    -o outputs/example1_color.jpg \
    --model-path $MODEL_PATH \
    --seed 42

echo ""
echo "=== Example 2: Object addition (CARD: medium complexity, Nr=8, r=4) ==="
python physedit.py \
    -i examples/input_scene.jpg \
    -p "Add a golden retriever sitting next to the person" \
    -o outputs/example2_add.jpg \
    --model-path $MODEL_PATH \
    --seed 42

echo ""
echo "=== Example 3: Physical action (CARD: high complexity, Nr=15, r=8) ==="
python physedit.py \
    -i examples/input_robot.jpg \
    -p "The robot picks up the red cup from the table" \
    -o outputs/example3_action.jpg \
    --model-path $MODEL_PATH \
    --seed 42

echo ""
echo "=== Example 4: Comparison with baseline ChronoEdit ==="
python physedit.py \
    -i examples/input_color.jpg \
    -p "Change the hat to a bright red cap" \
    -o outputs/example4_baseline.jpg \
    --model-path $MODEL_PATH \
    --disable-sce \
    --seed 42

echo ""
echo "All examples completed. Check outputs/ directory."
