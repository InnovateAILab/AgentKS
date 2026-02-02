FROM python:3.11-slim
WORKDIR /app
# copy requirements and install (includes supervisor)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# copy the full backend app into the image so both ASGI apps are available
COPY . /app

# expose both service ports
EXPOSE 8000
EXPOSE 4000

# Use a startup script that runs alembic migrations and then supervisord
CMD ["/bin/sh", "/app/startup.sh"]