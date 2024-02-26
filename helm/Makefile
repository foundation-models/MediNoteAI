namespace=default

deploy:
	helm uninstall generative-ai	
	helm upgrade --install generative-ai generative-ai -f generative-ai/ci/desktop.yaml
PHONY: deploy

deploy-image-qa:
	@if helm list --filter '^generative-ai' | grep -q 'generative-ai'; then \
		echo "Uninstalling 'generative-ai'..."; \
		helm uninstall generative-ai; \
	else \
		echo "Release 'generative-ai' does not exist. No action taken."; \
	fi	
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

deploy-sandbox:
	helm upgrade --install sandbox generative-ai -f generative-ai/ci/sandbox.yaml
	argo delete @latest -n ${namespace}
	argo submit  -n ${namespace} --watch data-curation.yaml
	argo list -n ${namespace}
PHONY: deploy-sandbox

deploy-aicluster:
	helm upgrade --install sandbox generative-ai -f generative-ai/ci/aicluster.yaml
PHONY: deploy-aicluster

deploy-workspace:
	helm upgrade --install workspace generative-ai -f generative-ai/ci/workspace.yaml
	argo delete @latest -n ai
	argo submit  -n ai --watch data-curation-dev.yaml
	argo list -n ai
PHONY: deploy-workspace

delete-latest-workflow:
	argo delete @latest -n ${namespace}
PHONY: delete-latest-workflow

helm-template:
	helm template llama generative-ai -f generative-ai/ci/desktop.yaml > out.yaml
PHONY: helm-template

deploy-service:
	kubectl apply -f service.yaml
PHONY: deploy-service

force-delete:
	kubectl delete pods  --grace-period=0 --force
PHONY: force-delete