
#!/bin/bash
POD_HOST=${POD_HOST:-213.173.96.45}
POD_PORT=${POD_PORT:-11120}
echo "Tunnelling localhost:8000 -> RunPod:8000  (GPU 0)"
echo "Tunnelling localhost:8002 -> RunPod:8002  (GPU 1)"
ssh -N \
  -L 8000:localhost:8000 \
  -L 8002:localhost:8002 \
  root@$POD_HOST -p $POD_PORT -i ~/.ssh/id_ed25519