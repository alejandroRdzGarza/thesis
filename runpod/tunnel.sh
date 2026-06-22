
#!/bin/bash
POD_HOST=${POD_HOST:-213.192.2.117}
POD_PORT=${POD_PORT:-40077}
echo "Tunnelling localhost:8000 -> RunPod:8000"
ssh -N -L 8000:localhost:8000 root@$POD_HOST -p $POD_PORT -i ~/.ssh/id_ed25519