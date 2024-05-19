namespace=default

desktop:
	-helm uninstall generative-ai	
	helm upgrade --install generative-ai generative-ai -f generative-ai/ci/desktop.yaml
PHONY: desktop


yellow:	
	helm upgrade --install yellow generative-ai -f generative-ai/ci/yellow.yaml
PHONY: yellow

docker-engine:	
	helm upgrade --install docker-engine generative-ai -f generative-ai/ci/docker-engine.yaml
PHONY: docker-engine


workspace:
	helm upgrade --install workspace --namespace ai generative-ai -f generative-ai/ci/workspace.yaml 
PHONY: workspace

workspace-gpu:
	helm upgrade --install workspace-gpu --namespace ai generative-ai -f generative-ai/ci/workspace-gpu.yaml 
PHONY: workspace-gpu


basic:
	-helm uninstall generative-ai	
	helm upgrade --install generative-ai generative-ai -f generative-ai/ci/basic.yaml
PHONY: basic

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
	# argo delete @latest -n ${namespace}
	# argo submit  -n ${namespace} --watch data-curation.yaml
	# argo list -n ${namespace}
PHONY: deploy-sandbox


deploy-workflow:
	helm upgrade --install workflow generative-ai -f generative-ai/ci/workflow.yaml
PHONY: deploy-workflow


deploy-encoder:
	helm uninstall encoder
	helm upgrade --install encoder generative-ai -f generative-ai/ci/encoder.yaml
PHONY: deploy-encoder



deploy-weaviate:
# helm repo add weaviate https://weaviate.github.io/weaviate-helm
	helm upgrade --install \
	"weaviate" \
	weaviate/weaviate \
	-f weaviate/ci/default.yaml
PHONY: deploy-weaviate

deploy-aicluster:
	helm upgrade --install sandbox generative-ai -f generative-ai/ci/aicluster.yaml
PHONY: deploy-aicluster

deploy-workspace:
	# helm upgrade --install workspace generative-ai -f generative-ai/ci/workspace.yaml
	argo delete @latest -n ai
	argo submit  -n ai --watch data-curation-dev.yaml
	argo list -n ai
PHONY: deploy-workspace

deploy-autogen:
	helm upgrade --install autogen generative-ai -f generative-ai/ci/autogen.yaml
PHONY: deploy-autogen

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

speacial-deploy:
	cd ../../milvus-helm/charts && \
	helm upgrade --install milvus milvus -f ../../workspace/ai-dev-argo-apps/helm/milvus/ci/aicluster.yaml
.PHONY: speacial-deploy

deletepod:
	kubectl delete pods sqlcoder-generative-ai-cbb4646d8-9ftzn --grace-period=0 --force
.PHONY: deletepod

deploy-tes-gpu:
	kubectl apply -f test-gpu.yaml
PHONY: deploy-test-gpu

testgpu:
	microk8s kubectl logs -n gpu-operator-resources -lapp=nvidia-operator-validator -c nvidia-operator-validator
.PHONY: testgpu

dask-operator:
	helm install --repo https://helm.dask.org --create-namespace -n dask-operator --generate-name dask-kubernetes-operator
.PHONY: dask-operator

dask:
	helm uninstall dask --ignore-not-found
	helm upgrade --install dask dask-charts -f dask-charts/dask-cpu-values.yaml 
.PHONY: dask

