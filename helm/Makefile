

deploy:
	helm upgrade --install generative-ai generative-ai -f generative-ai/ci/desktop.yaml
PHONY: deploy

deploy-image-qa:
	helm uninstall generative-ai	
	helm upgrade --install generative-ai generative-ai -f generative-ai/ci/image-qa.yaml
PHONY: deploy-image-qa