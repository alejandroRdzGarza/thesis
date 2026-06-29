
#!/bin/bash
POD_HOST=${POD_HOST:-157.157.221.29}
POD_PORT=${POD_PORT:-25494}
echo "Tunnelling localhost:8000 -> RunPod:8000"
ssh -N -L 8000:localhost:8000 root@$POD_HOST -p $POD_PORT -i ~/.ssh/id_ed25519