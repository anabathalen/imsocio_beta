# Installation and Setup Guide for imsocio

## Quick Start

### 1. Install Python

Make sure you have Python 3.8 or later installed. Check your version:

```bash
python --version
```

### 2. Clone or Download the Repository

```bash
git clone https://github.com/anabathalen/imsocio.git
cd imsocio
```

Or download and extract the ZIP file, then navigate to the folder.

### 3. Create a Virtual Environment (Recommended)

#### On Windows:
```powershell
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install the Package

```bash
pip install -e .
```

This installs imsocio in "editable" mode, so you can make changes to the code if needed.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

## Manual Installation

If you prefer to install dependencies separately:

```bash
pip install -r requirements.txt
```

## Troubleshooting

### "Python not found"
- Make sure Python is installed and added to your PATH
- Try using `python3` instead of `python` on macOS/Linux

### "streamlit: command not found"
- Make sure your virtual environment is activated
- Try: `python -m streamlit run app.py`

### Import Errors
- Make sure you've installed the package: `pip install -e .`
- Check that you're in the correct directory
- Verify your virtual environment is activated

### Port Already in Use
If port 8501 is already in use, specify a different port:

```bash
streamlit run app.py --server.port 8502
```

## Development Installation

For development with testing tools:

```bash
pip install -e ".[dev]"
```

This installs additional tools like pytest, black, and flake8.

## Verifying Installation

Test that everything is installed correctly:

```python
python -c "import imsocio; print('imsocio version:', imsocio.__version__)"
```

## Updating

To update to the latest version:

```bash
git pull origin main
pip install -e . --upgrade
```

## Uninstalling

```bash
pip uninstall imsocio
```

## System Requirements

- **Python**: 3.8 or later
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for installation
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

## Dependencies

Core dependencies are automatically installed:
- pandas (>=1.3.0)
- numpy (>=1.20.0)
- scipy (>=1.7.0)
- matplotlib (>=3.4.0)
- plotly (>=5.0.0)
- streamlit (>=1.20.0)
- scikit-learn (>=0.24.0)

## Getting Help

- **Documentation**: See `README.md` for detailed usage
- **Issues**: Report bugs at https://github.com/anabathalen/imsocio/issues
- **Email**: your.email@example.com
