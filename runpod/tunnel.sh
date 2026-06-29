
#!/bin/bash
POD_HOST=${POD_HOST:-195.26.233.70}
POD_PORT=${POD_PORT:-24862}
echo "Tunnelling localhost:8000 -> RunPod:8000"
ssh -N -L 8000:localhost:8000 root@$POD_HOST -p $POD_PORT -i ~/.ssh/id_ed25519