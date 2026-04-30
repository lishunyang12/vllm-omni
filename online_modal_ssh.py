"""
Sets up an SSH server in a Modal container with a vllm-omni dev environment.
After running with `modal run online_modal_ssh.py`, connect with:
  ssh -p <port> root@<host>
or add an entry to ~/.ssh/config for VSCode Remote-SSH.
"""
import modal
import threading
import socket
import subprocess
import time
import os

ssh_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "UTC"})
    .apt_install(
        "wget", "git", "sox", "libsox-fmt-all", "jq", "curl",
        "openssh-server", "pkg-config", "build-essential",
    )
    .run_commands(
        "mkdir -p /run/sshd",
        "mkdir -p /root/.ssh",
    )
    .add_local_file(ssh_key_path, "/root/.ssh/authorized_keys", copy=True)
    .run_commands("chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys")
    .run_commands(
        "echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config",
        "echo 'PubkeyAuthentication yes' >> /etc/ssh/sshd_config",
        "echo 'PasswordAuthentication no' >> /etc/ssh/sshd_config",
    )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .run_commands("uv pip install --system vllm==0.19.0 --torch-backend cu130")
    .run_commands("git clone https://github.com/ArtificialRay/vllm-omni.git /vllm-omni")
    .run_commands("cd /vllm-omni && uv pip install --system -e '.[dev]'")
    .run_commands(
        "uv pip uninstall --system opencv-python || true",
        "uv pip install --system --reinstall opencv-python-headless",
        "uv pip install --system 'nvidia-modelopt[all]'",
        "uv pip install --system lpips",
        "uv pip install --system ftfy"
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App("vllm-omni-ssh", image=image, volumes={"/root/.cache/huggingface": hf_cache_vol})


def wait_for_port(host, port, q):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 22), timeout=30.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= 30.0:
                raise TimeoutError("Waited too long for port 22 to accept connections") from exc
    q.put((host, port))


@app.function(gpu="H100", timeout=3600 * 24)
def launch_ssh(q):
    with modal.forward(22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        threading.Thread(target=wait_for_port, args=(host, port, q)).start()
        subprocess.run(["/usr/sbin/sshd", "-D"])


@app.local_entrypoint()
def main():
    with modal.Queue.ephemeral() as q:
        launch_ssh.spawn(q)
        host, port = q.get()
        print(f"SSH server running at {host}:{port}")
        print(f"Connect with: ssh -p {port} root@{host}")
        print(f"Or add to ~/.ssh/config:")
        print(f"  Host vllm-omni-modal")
        print(f"    HostName {host}")
        print(f"    Port {port}")
        print(f"    User root")
        print(f"    IdentityFile ~/.ssh/id_rsa")
        print(f"  vllm-omni source: /vllm-omni")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
