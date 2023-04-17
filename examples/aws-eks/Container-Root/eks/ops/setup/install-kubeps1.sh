#!/bin/bash


curl -L -o ~/kube-ps1.sh https://github.com/jonmosco/kube-ps1/raw/master/kube-ps1.sh

cat << EOF >> ~/.bashrc
alias ll='ls -alh --color=auto'
alias kon='touch ~/.kubeon; source ~/.bashrc'
alias koff='rm -f ~/.kubeon; source ~/.bashrc'
alias kctl='kubectl'
alias kc='kubectx'
alias kn='kubens'
alias kt='kubetail'
alias ks='kubectl node-shell'
alias nv='eks-node-viewer'
alias tx='torchx'
alias wp='watch-pods.sh'
alias wn='watch-nodes.sh'
alias wnt='watch-node-types.sh'
alias lp='pods-list.sh'
alias dp='pod-describe.sh'
alias ln='nodes-list.sh'
alias lnt='nodes-types-list.sh'
alias pe='pod-exec.sh'
alias pl='pod-logs.sh'

if [ -f ~/.kubeon ]; then
        source ~/kube-ps1.sh
        PS1='[\u@\h \W \$(kube_ps1)]\$ '
fi

export TERM=xterm-256color

export PATH=$PATH:/root/go/bin:/eks/ops:.

EOF


