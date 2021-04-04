#!/usr/bin/env bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 images_path annotations.json"
    exit 1
fi

tmp="/tmp/odtk"

tests=(
    "odtk train ${tmp}/model.pth --images $1 --annotations $2 --max-size 640 --override --iters 100 --backbone ResNet18FPN ResNet50FPN"
    "odtk train ${tmp}/model.pth --images $1 --annotations $2 --max-size 640 --override --iters 100"
    "odtk train ${tmp}/model.pth --fine-tune ${tmp}/model.pth --images $1 --annotations $2 --max-size 640 --override --iters 100"
    "odtk infer ${tmp}/model.pth --images ${tmp}/test_images --max-size 640"
    "odtk export ${tmp}/model.pth ${tmp}/engine.plan --size 640"
    "odtk infer ${tmp}/engine.plan --images ${tmp}/test_images --max-size 640"
)

start=`date +%s`

# Prepare small image folder for inference
if [ ! -d ${tmp}/test_images ]; then
    mkdir -p ${tmp}/test_images
    cp $(find $1 | tail -n 10) ${tmp}/test_images
fi

# Run all tests
for test in "${tests[@]}"; do
    echo "Running \"${test}\""
    ${test}
    if [ $? -ne 0 ]; then
        echo "Test failed!"
        exit 1
    fi
done

end=`date +%s`

echo "All test succeeded in $((end-start)) seconds!"