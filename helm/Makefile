namespace=default

deploy:
	helm uninstall generative-ai	
	helm upgrade --install generative-ai generative-ai -f generative-ai/ci/desktop.yaml
PHONY: deploy

deploy-image-qa:
	helm uninstall generative-ai	
	helm upgrade --install generative-ai generative-ai -f generative-ai/ci/image-qa.yaml
PHONY: deploy-image-qa

install-argo:
	curl -sLO https://github.com/argoproj/argo-workflows/releases/download/v3.5.4/argo-linux-amd64.gz
	gunzip argo-linux-amd64.gz
	chmod +x argo-linux-amd64
	sudo mv ./argo-linux-amd64 /usr/local/bin/argo
	argo version
PHONY: install-argo

install-argo-workflow:
	kubectl create namespace ${namespace} --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -n ${namespace} -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.4/quick-start-minimal.yaml
PHONY: install-argo-workflow

port-forward-argo:
	kubectl port-forward svc/argo-server -n argo 2746:2746
PHONY: port-forward-argo

lint:
	argo lint data-curation.yaml
PHONY: lint

submit-workflow:
	argo delete @latest -n ${namespace}
	argo submit  -n ${namespace} --watch data-curation.yaml
	argo list -n ${namespace}
PHONY: submit-workflow

delete-workflow:
	argo delete @latest -n ${namespace}
PHONY: delete-workflow

helm-template:
	helm template llama generative-ai -f generative-ai/ci/desktop.yaml > out.yaml
PHONY: helm-template

deploy-service:
	kubectl apply -f service.yaml
PHONY: deploy-service