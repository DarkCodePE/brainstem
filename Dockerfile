FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml LICENSE README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

VOLUME /vault
EXPOSE 8765

ENTRYPOINT ["brainstem"]
CMD ["mcp", "--root", "/vault"]
