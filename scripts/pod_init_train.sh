#!/usr/bin/env bash
# Minimal sshd init for a train-only pod (no llama.cpp build).
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends openssh-server ca-certificates curl git python3 python3-pip rsync
mkdir -p /run/sshd /root/.ssh /workspace/logs
printf '%s\n' 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOQ0s1PJuMfLm9pJNuPJSFzNnA/hmp8uO7m88Yg5O2Fd llama-nexus-runpod-automation' > /root/.ssh/authorized_keys
chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
sed -i 's/^#\?UsePAM.*/UsePAM no/' /etc/ssh/sshd_config
grep -q '^UsePAM' /etc/ssh/sshd_config || echo 'UsePAM no' >> /etc/ssh/sshd_config
exec /usr/sbin/sshd -D -e
