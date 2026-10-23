# Regenerates requirements.lock from the working native venv. Run inside WSL.
# --path: only the venv's own site-packages. The ROCDXG deb adds a .pth that
# puts /opt/rocm-wsl/share/amd_smi on sys.path, and a plain freeze reports that
# copy (amdsmi 1.2.2) instead of the wheel the venv installed.
source ~/vllm-env/bin/activate
pip freeze --path "$VIRTUAL_ENV/lib/python3.12/site-packages" > "$(dirname "$0")/requirements.lock"
