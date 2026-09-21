#!/usr/bin/env bash

INPUT="videos/3-view/angled-no-light/side1.mp4"
OUTPUT="videos/3-view/angled-no-light/side1-crop.mp4"

CROP="255:80:195:200" # WIDTH:HEIGHT:X:Y

ffmpeg -i "$INPUT" \
    -vf "crop=$CROP" \
    -c:v libx264 -crf 18 -preset fast \
    "$OUTPUT"