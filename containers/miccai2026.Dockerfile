FROM fuzzy-pytorch:sr
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
# This is CRITICAL. It tells 'uv run' to use the existing /usr/local environment
# (where your custom PyTorch is) instead of creating a new .venv.
ENV UV_PROJECT_ENVIRONMENT="/usr/local"
# Prevents uv from downloading python builds, forcing it to use the base image's
ENV UV_PYTHON_DOWNLOADS="never"

COPY pyproject.toml uv.lock ./

RUN : \
    && uv export --frozen --no-hashes --format=requirements-txt > requirements.txt \
    && grep -vE "^(torch)" requirements.txt > filtered_reqs.txt \
    && uv pip install --system -r filtered_reqs.txt \
    && :

ENTRYPOINT ["uv run"]