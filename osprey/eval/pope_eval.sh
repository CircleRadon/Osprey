
for type in random popular adversarial
    do 
        python pope_eval.py --model path/to/osprey-chat-7b \
        --img path/to/coco_imgs --json pope/coco_pope_${type}.json \
        --answer pope/coco_pope_${type}_answers.json
done

for type in random popular adversarial
    do 
        echo "Evaluating pope on ${type} data..."
        python pope/evaluate.py --ans-file pope/coco_pope_${type}_answers.json  \
        --label-file pope/coco_pope_${type}.json
done