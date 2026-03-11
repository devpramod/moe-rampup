set -euo pipefail

# 1) Tools
sudo apt-get update
sudo apt-get install -y build-essential numactl wget

# 2) Download STREAM source
wget -O /tmp/stream.c https://www.cs.virginia.edu/stream/FTP/Code/stream.c

# 3) Build (array size large enough to exceed cache)
if ! gcc -O3 -fopenmp -march=native -mcmodel=medium -no-pie \
  -DSTREAM_ARRAY_SIZE=200000000 \
  -DNTIMES=20 \
  /tmp/stream.c -o /tmp/stream; then
  echo "Primary STREAM build failed, retrying with smaller array size..."
  gcc -O3 -fopenmp -march=native -no-pie \
    -DSTREAM_ARRAY_SIZE=60000000 \
    -DNTIMES=20 \
    /tmp/stream.c -o /tmp/stream
fi

# 4) Quick smoke test
OMP_NUM_THREADS=1 /tmp/stream | tail -n 20