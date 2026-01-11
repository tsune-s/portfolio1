# Contributing to Data Analysis Portfolio

Thank you for your interest in contributing to this portfolio project!

## Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd portfolio
```

2. Install dependencies:
```bash
uv sync
```

3. Generate sample data:
```bash
uv run python scripts/generate_sample_data.py
```

## Code Style

This project follows these coding standards:

- **Formatting**: Code is formatted using Black
- **Linting**: Code is linted using Ruff
- **Type Hints**: Use type hints for function signatures
- **Docstrings**: Use Google-style docstrings

### Running Code Quality Checks

```bash
# Format code
uv run black src/ tests/ scripts/

# Lint code
uv run ruff check src/ tests/ scripts/

# Run tests
uv run pytest
```

## Project Structure

```
portfolio/
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   ├── models/           # Machine learning models
│   ├── visualization/    # Plotting utilities
│   └── utils/            # Helper functions
├── scripts/              # Automation scripts
├── tests/                # Unit tests
├── dashboards/           # Streamlit apps
└── notebooks/            # Jupyter notebooks
```

## Adding New Features

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Implement your feature with tests

3. Ensure all tests pass:
```bash
uv run pytest
```

4. Format and lint your code:
```bash
uv run black .
uv run ruff check .
```

5. Commit your changes:
```bash
git commit -m "Add feature: description"
```

6. Push and create a pull request

## Testing

- Write unit tests for all new functions
- Aim for high test coverage
- Use pytest fixtures for common test data
- Mock external dependencies

Example test:
```python
def test_my_function():
    result = my_function(input_data)
    assert result == expected_output
```

## Documentation

- Update README.md if adding major features
- Add docstrings to all public functions
- Include usage examples in docstrings

## Questions?

If you have questions or need help, please open an issue on GitHub.
