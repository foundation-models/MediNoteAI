ollama:
	cd ollama-webui && \
	rm -f build && \
	ln -s /app/build build && \
	cd backend && \
	export OLLAMA_API_BASE_URL=llama-generative-ai:5000/api && \
	uvicorn main:app --host 0.0.0.0 --port 8888 --forwarded-allow-ips '*'
.PHONY: ollama

autogen:
	cd autogen/samples/apps/autogen-studio && \
	mv frontend frontend.bak && \
	ln -s /app/frontend frontend && \
	cd autogenstudio && \
	mv web web.bak && \
	ln -s /app/autogenstudio/web web && \
	cd .. && \
	uvicorn autogenstudio.web.app:app --host 0.0.0.0 --port 8888 --workers 1
.PHONY: autogen

end-autogen:
	cd autogen/samples/apps/autogen-studio && \
	rm frontend && \
	mv frontend.bak frontend && \
	cd autogenstudio && \
	rm web && \
	mv web.bak web
.PHONY: end-autogen

webui:
	cd ../src/django-chatgpt-chatbot && \
	python manage.py runserver 0.0.0.0:8888
.PHONY: webui


create_superuser:
	@echo "Creating superuser..."
	@cd ../src/django-chatgpt-chatbot && python manage.py createsuperuser
	@echo "Done."


update_tables:
	@echo "Updating tables..."
	@cd ../src/django-chatgpt-chatbot && python manage.py makemigrations
	@cd ../src/django-chatgpt-chatbot && python manage.py migrate
	@echo "Done."
