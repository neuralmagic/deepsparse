#!/bin/bash

# Install kubectl
# Reference: https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/#install-kubectl-binary-with-curl-on-linux 
curl -Lo kubectl https://dl.k8s.io/release/v1.25.4/bin/linux/amd64/kubectl
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin
kubectl version --client --short

# Install bash completion
# Reference: https://v1-25.docs.kubernetes.io/docs/tasks/tools/install-kubectl-linux/#enable-shell-autocompletion
echo 'source /usr/share/bash-completion/bash_completion' >> /root/.bashrc
echo 'source <(kubectl completion bash)' >> /root/.bashrc
echo 'alias k=kubectl' >> /root/.bashrc
echo 'complete -o default -F __start_kubectl k' >> /root/.bashrc

