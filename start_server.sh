#!/bin/bash
# Start Blender Server for Monty Visual Agent
# 
# Usage:
#   ./start_server.sh [options]
#
# Options:
#   --port PORT     Server port (default: 5555)
#   --gpu           Enable GPU rendering
#   --width WIDTH   Render width (default: 320)
#   --height HEIGHT Render height (default: 240)
#   --camera NAME   Camera name (default: Camera)
#   --help          Show this help

set -e

cd "$(dirname "$0")"

# Default values
PORT=5555
WIDTH=320
HEIGHT=240
CAMERA="Camera"
GPU_FLAG=""
SCENE="blender/simulation_scene.blend"
SERVER_SCRIPT="src/server/blender_server.py"

# Blender paths for different platforms
if [[ "$OSTYPE" == "darwin"* ]]; then
    BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    BLENDER="blender"
else
    BLENDER="blender"
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --gpu)
            GPU_FLAG="--gpu"
            shift
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --camera)
            CAMERA="$2"
            shift 2
            ;;
        --scene)
            SCENE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./start_server.sh [options]"
            echo ""
            echo "Options:"
            echo "  --port PORT     Server port (default: 5555)"
            echo "  --gpu           Enable GPU rendering"
            echo "  --width WIDTH   Render width (default: 320)"
            echo "  --height HEIGHT Render height (default: 240)"
            echo "  --camera NAME   Camera name (default: Camera)"
            echo "  --scene FILE    Blender scene file (default: blender/simulation_scene.blend)"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if Blender exists
if ! command -v "$BLENDER" &> /dev/null && [ ! -f "$BLENDER" ]; then
    echo "Error: Blender not found at: $BLENDER"
    echo ""
    echo "Please install Blender or update the path in this script."
    echo ""
    echo "macOS: brew install --cask blender"
    echo "Linux: sudo apt install blender"
    exit 1
fi

# Check if scene file exists
if [ ! -f "$SCENE" ]; then
    echo "Error: Scene file not found: $SCENE"
    exit 1
fi

# Check if server script exists
if [ ! -f "$SERVER_SCRIPT" ]; then
    echo "Error: Server script not found: $SERVER_SCRIPT"
    exit 1
fi

echo "=============================================="
echo "Blender+Monty Server"
echo "=============================================="
echo ""
echo "Scene:      $SCENE"
echo "Port:       $PORT"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Camera:     $CAMERA"
echo "GPU:        ${GPU_FLAG:-disabled}"
echo ""
echo "Starting server..."
echo ""

# Start Blender with server script
"$BLENDER" \
    -b "$SCENE" \
    --python "$SERVER_SCRIPT" \
    -- \
    --host 127.0.0.1 \
    --port "$PORT" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --camera "$CAMERA" \
    $GPU_FLAG