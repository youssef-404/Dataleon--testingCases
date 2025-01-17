# Dataleon--testingCases

This project uses the detection transformer model for detecting tables in invoice and bank document images.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/youssef-404/Dataleon--testingCases.git
   cd Dataleon--testingCases
   ```

2. Build the Docker image:
   ```
   docker build -t table-detector .
   ```

3. Run the tests:
   ```
   docker run table-detector
   ```

## Usage

To use the `TableDetector` class in your own code:

```python
from table_detector import TableDetector

detector = TableDetector()
results = detector.detect("path/to/your/image.jpg")
print(results)
```

## Testing

The `tests` directory contains various test cases for different scenarios. To run the tests outside of Docker:

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run pytest:
   ```
   pytest -v
   ```

## Project Structure

- `src/table_detector.py`: Contains the main `TableDetector` class.
- `tests/test_table_detector.py`: Contains pytest test cases.
- `Dockerfile`: Sets up the Docker environment for running tests.
- `requirements.txt`: Lists all Python dependencies.

