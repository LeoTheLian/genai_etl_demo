# GenAI ETL Demo

This repository demonstrates the application of agentic AI to accelerate and automate data ingestion pipeline development. The system uses specialized AI agents to transform natural language requirements into executable ETL code, perform data profiling, and generate comprehensive data quality tests.

## Project Overview

The demo implements an end-to-end ETL pipeline for processing credit card transaction data for fraud detection analytics. The pipeline transforms raw transaction data from a simulated card payment vendor feed into a clean, analytics-ready format.

### Business Context

Organizations need reliable, scalable ingestion pipelines to process raw credit card transaction data and prepare it for fraud detection models and analytical workloads. This demo showcases how AI agents can automate the traditionally manual process of ETL development.

### Key Features

- **AI-Driven Development**: LLM-powered agents automatically generate ETL code from requirements
- **Automated Testing**: AI-generated data quality tests validate pipeline outputs
- **Modular Architecture**: Specialized agents handle different aspects of ETL development
- **Framework Flexibility**: Supports both pandas and PySpark execution frameworks
- **Comprehensive Validation**: Built-in data profiling, schema validation, and quality checks

## Architecture

The system consists of specialized AI agents that work together to automate ETL development:

### Core Agents

#### 1. Analyst Agent (`agents/analyst_agent.py`)
- **Purpose**: Analyzes requirements documents and source CSV data to produce source-to-target mappings and a data dictionary in a single consolidated call
- **Input**: Requirements document (`data/raw/requirements_document.txt`), source CSV (`data/raw/sample_transactions.csv`)
- **Output**: Source-to-target mapping JSON (`outputs/source_to_target_mapping.json`), data dictionary JSON (`outputs/source_data_dictionary.json`)
- **Capabilities**:
  - Extracts schema definitions from unstructured text
  - Identifies column mappings and transformations
  - Profiles source data for column statistics and metadata
  - Supports both LLM-enhanced and rule-based parsing modes

#### 2. Developer Agent (`agents/developer_agent.py`)
- **Purpose**: Generates executable ETL pipeline code from source-to-target mappings
- **Input**: Source-to-target mapping JSON
- **Output**: Executable Python script (`outputs/generated_pipeline.py`)
- **Capabilities**:
  - LLM-only code generation using specialized prompts
  - Framework-agnostic pipeline creation (pandas/PySpark)
  - Integration with ETL utility functions
  - Automatic validation report generation
  - Feedback-aware iterative regeneration using test results and versioned pipeline artifacts

#### 3. Testing Agent (`agents/testing_agent.py`)
- **Purpose**: Generates and executes data quality tests against processed data
- **Input**: Requirements, source data, and processed data
- **Output**: Test report (`data/processed/test_report.txt`) and generated tests (`outputs/generated_tests.py`)
- **Capabilities**:
  - LLM-driven test case generation from requirements
  - Automated test execution and reporting
  - Schema validation and transformation verification

#### 4. Orchestrator (`agents/orchestrator.py`)
- **Purpose**: Coordinates the entire ETL development workflow
- **Capabilities**:
  - Sequential execution of all agents
  - Pipeline execution and validation
  - Feedback loop orchestration: repeat pipeline generation, execution, and testing until all tests pass
  - Comprehensive status reporting

### Supporting Components

#### ETL Utilities (`agents/etl_utilities.py`)
A comprehensive library of reusable data transformation functions:
- Data loading and saving
- Column renaming and type casting
- Filtering and validation
- Derived column calculations
- Standardization functions

#### LLM Client (`agents/llm_client.py`)
OpenAI API integration with:
- JSON and text response handling
- Configurable model selection
- Error handling and retries
- Environment variable configuration

## Workflow

The complete ETL development process follows these steps:

1. **Requirements Analysis**
   - Analyze requirements document
   - Generate source-to-target mapping
   - Extract transformation rules

2. **Pipeline Generation**
   - Create executable ETL code
   - Apply transformations and validations
   - Generate validation reports

3. **Pipeline Execution**
   - Run generated pipeline on source data
   - Produce processed output
   - Create validation summary

4. **Quality Testing**
   - Generate comprehensive test suite
   - Execute tests against processed data
   - Produce detailed test reports

## Data Flow

```
Requirements Document
        ↓
Source-to-Target Mapping
        ↓
Generated Pipeline Code
        ↓
Processed Data + Validation Report
        ↓
Test Results + Quality Report
```

## Sample Data

The demo uses simulated credit card transaction data:
- **Source**: `data/raw/sample_transactions.csv` (1,000 transactions)
- **Schema**: 25 columns including transaction details, account info, merchant data
- **Target**: `data/processed/fraud_transactions.csv` (32 columns with derived fields)
- **Fraud Distribution**: ~3% fraudulent transactions

## Usage

### Prerequisites

1. **Python Environment**: Python 3.8+
2. **Dependencies**: Install via `pip install -r requirements.txt`
3. **OpenAI API**: Set `OPENAI_API_KEY` environment variable

### Quick Start

```bash
# Run complete workflow
python agents/orchestrator.py

# Run with PySpark framework
python agents/orchestrator.py --framework pyspark

# Run iterative improvement until tests pass
python agents/orchestrator.py --iterate --max-iterations 5

# Generate artifacts without execution
python agents/orchestrator.py --skip-run
```

### Individual Agent Usage

```bash
# Analyze requirements and data
python agents/analyst_agent.py

# Generate pipeline only
python agents/developer_agent.py

# Run tests only
python agents/testing_agent.py
```

### Environment Setup

```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_api_key_here

# Or create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Output Artifacts

The workflow generates several artifacts:

- `outputs/source_to_target_mapping.json`: Structured mapping definition
- `outputs/generated_pipeline.py`: Executable ETL code
- `outputs/generated_tests.py`: Generated test suite
- `data/processed/fraud_transactions.csv`: Cleaned output data
- `data/processed/validation_report.txt`: Processing summary
- `data/processed/test_report.txt`: Quality test results
- `outputs/agent_activity_log.json`: Persistent activity log for agent events, iteration summaries, and retrievable prompts/outputs
- `outputs/generated_pipeline_iteration_{n}.py`, `data/processed/test_report_iteration_{n}.txt`, `outputs/test_summary_iteration_{n}.json`: Versioned iteration artifacts when run with `--iterate`

## Configuration

### LLM Settings
Configure via environment variables:
- `OPENAI_API_KEY`: Required API key
- `OPENAI_BASE_URL`: Optional custom endpoint (default: OpenAI)
- `OPENAI_MODEL`: Optional model selection (default: gpt-4o-mini)

### Pipeline Options
- **Framework**: `pandas` (default) or `pyspark`
- **Execution Mode**: Generate-only or execute pipeline
- **File Paths**: Customizable input/output locations

## Validation & Testing

The system includes comprehensive validation:

- **Schema Validation**: Ensures all required columns present
- **Type Checking**: Validates data types match specifications
- **Transformation Verification**: Confirms business rules applied
- **Data Quality Tests**: Automated checks for data integrity
- **Processing Reports**: Detailed summaries of transformations

## Example Results

Recent execution processed 1,000 input records:
- **Output Records**: 965 (35 filtered due to validation rules)
- **Fraud Distribution**: 935 legitimate, 30 fraudulent
- **Amount Categories**: 241 low, 482 medium, 193 high, 49 outlier
- **Test Coverage**: 7 automated validation checks

## Architecture Benefits

- **Speed**: Reduces ETL development from weeks to minutes
- **Consistency**: Eliminates human error in mapping implementation
- **Maintainability**: Self-documenting code with comprehensive tests
- **Scalability**: Framework-agnostic design supports various platforms
- **Auditability**: Complete traceability from requirements to execution

## Future Enhancements

- Multi-source data integration
- Advanced transformation patterns
- Real-time pipeline execution
- Cloud-native deployment options
- Enhanced error handling and recovery

## Contributing

This is a demonstration project showcasing AI-driven ETL development. For questions or contributions, please refer to the agent implementations and utility functions.
