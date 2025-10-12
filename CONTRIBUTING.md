# Contributing to Wood Identification

Thank you for your interest in contributing to the Wood Identification project! This document provides guidelines and information for contributors.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior vs actual behavior
- Your environment (OS, Python version, package versions)
- Any relevant error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- A clear description of the enhancement
- The motivation/use case for the feature
- Examples of how it would work
- Any potential drawbacks or considerations

### Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the code style guidelines below
   - Add tests for new functionality
   - Update documentation as needed

3. **Validate your changes**
   ```bash
   # Check syntax
   python validate_structure.py
   
   # Run tests (if available)
   pytest tests/
   
   # Check code style
   black wood_identification/
   flake8 wood_identification/
   ```

4. **Commit your changes**
   - Write clear, concise commit messages
   - Reference relevant issues in your commits

5. **Push to your fork** and submit a pull request
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Wait for review**
   - Address any feedback from reviewers
   - Keep your PR up to date with the main branch

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/schwinger42/wood_identification.git
   cd wood_identification
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install package in editable mode
   ```

4. **Install development dependencies**
   ```bash
   pip install pytest black flake8 jupyter
   ```

## Code Style Guidelines

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use type hints where appropriate
- Maximum line length: 100 characters

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1, param2):
    """
    Brief description of function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When something goes wrong
    """
    pass
```

### Naming Conventions

- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Imports

Order imports as follows:
1. Standard library imports
2. Third-party imports
3. Local application imports

```python
import os
from pathlib import Path

import numpy as np
import torch

from wood_identification.data import RamanPreprocessor
```

## Testing Guidelines

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names: `test_preprocessor_handles_empty_spectrum`
- Test edge cases and error conditions
- Aim for high test coverage

### Test Structure

```python
def test_feature_name():
    """Test description"""
    # Arrange
    input_data = ...
    expected_output = ...
    
    # Act
    actual_output = function_under_test(input_data)
    
    # Assert
    assert actual_output == expected_output
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_preprocessing.py

# Run with coverage
pytest --cov=wood_identification
```

## Documentation Guidelines

### Code Documentation

- Document all public APIs
- Include usage examples in docstrings
- Keep documentation up to date with code changes

### README and Guides

- Use clear, concise language
- Include code examples
- Add screenshots where helpful
- Keep the table of contents updated

### Notebooks

- Clear explanations for each step
- Well-commented code
- Include output cells
- Test notebooks before committing

## Project Structure

When adding new features, maintain the project structure:

```
wood_identification/
├── wood_identification/      # Main package
│   ├── data/                # Data processing
│   ├── models/              # Model implementations
│   └── utils/               # Utilities
├── tests/                   # Test files
├── examples/                # Example notebooks
├── data/                    # Data directory
├── models/                  # Saved models
└── results/                 # Results and plots
```

## Areas for Contribution

### High Priority

- [ ] Add unit tests for all modules
- [ ] Improve documentation with more examples
- [ ] Add support for more data formats
- [ ] Implement additional preprocessing methods
- [ ] Add more visualization options

### Medium Priority

- [ ] Create Jupyter notebook tutorials
- [ ] Add support for hyperparameter tuning
- [ ] Implement cross-validation utilities
- [ ] Add model interpretability tools
- [ ] Create a web interface

### Nice to Have

- [ ] Add support for other spectroscopy types
- [ ] Implement transfer learning examples
- [ ] Add data augmentation techniques
- [ ] Create Docker container
- [ ] Add CI/CD pipeline

## Versioning

We use [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for new backwards-compatible functionality
- PATCH version for backwards-compatible bug fixes

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to:
- Open an issue for questions
- Join discussions in existing issues
- Reach out to maintainers

## Recognition

Contributors will be acknowledged in:
- The CONTRIBUTORS file
- Release notes
- The README (for significant contributions)

Thank you for contributing to Wood Identification!
