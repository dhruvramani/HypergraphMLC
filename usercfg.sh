kubectl config set-cluster kubernetes --server=https://$2:6443 --certificate-authority=/etc/kubernetes/pki/apiserver.crt

kubectl config set-credentials $1 --client-certificate=/storage/home/$1/.kube/key/$1.crt  --client-key=/storage/home/$1/.kube/key/$1.key

kubectl config set-context $1-context --cluster=kubernetes --namespace=$1-namespace --user=$1

kubectl config use-context $1-context
