#!/bin/bash

ffmpeg -framerate 7 -i movie/frame%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p example_match.mp4

