# Multi-Agent System for REALM-Bench Evaluation

Research codebase for the term paper "Knowledge Augmentation for Multi-Agent Orchestration"

## Overview

This project implements a sophisticated multi-agent orchestration framework built on LangGraph to evaluate the impact of knowledge augmentation (Retrieval-Augmented Generation) on agent coordination and planning performance. The system decomposes complex planning and optimization tasks into subtasks, distributes them among specialized agents, and measures performance across various configurations.

### Key Features

- **LangGraph-based Orchestration**: State machine workflow with planning, execution, monitoring, and replanning capabilities
- **Multi-Agent Coordination**: Asynchronous parallel execution with dependency management and deadlock prevention
- **Knowledge Augmentation**: Optional RAG integration for enhanced planning and execution
- **REALM-Bench Integration**: Evaluation on standardized planning and optimization benchmarks
- **Specialized Agent Capabilities**: GENERAL, LOGISTICS, SCHEDULER, RESOURCE_MANAGER, OPTIMIZER, VALIDATOR, DATA_ANALYST
- **Docker-based Code Interpreter**: Sandboxed Python execution for computational tasks
- **Comprehensive Metrics**: Success rates, goal achievement, constraint satisfaction, token usage, planning quality

## Architecture

### Orchestration Workflow

```
Initialize → Plan → Execute ⇄ Monitor → [Replan/Finalize]
```

The orchestrator decomposes tasks into subtasks, assigns them to specialized worker agents, manages dependencies, monitors execution, and dynamically replans when needed.

### Directory Structure

```
Experiment/
├── src/
│   ├── agents/              # Multi-agent orchestration system
│   │   ├── orchestrator.py  # LangGraph workflow controller
│   │   ├── worker.py        # Worker agent implementation
│   │   ├── types.py         # Core data structures
│   │   ├── interaction_logger.py
│   │   └── tools/           # Agent tools (code interpreter, RAG)
│   ├── models/              # Unified model client API
│   │   └── registry.py
│   ├── realm/               # REALM-Bench integration
│   │   └── adapter.py
│   └── utils/
│       └── settings.py      # Configuration management
├── configs/
│   ├── base.yaml            # Base configuration
│   ├── models/              # Model-specific configs
│   └── experiments/         # Experiment configs (baseline, RAG)
├── scripts/
│   ├── run_experiments.py   # Experiment runner
│   └── analyze_metrics.py   # Metrics analysis
├── docker/
│   └── Dockerfile           # Code interpreter image
├── third_party/
│   └── realm_bench/         # REALM-Bench framework (submodule, needs to be cloned manually into this directory)
├── out/                     # Experiment outputs
├── requirements.txt
├── pyproject.toml
└── .env.template
```

## Setup & Installation

### Prerequisites

- Python 3.12+
- Docker (for code interpreter)
- Git

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Experiment
   ```

2. **Clone REALM-Bench submodule**
   ```bash
   mkdir -p third_party
   cd third_party
   git clone <realm-bench-url> realm_bench
   cd ..
   ```

3. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or using Poetry:
   ```bash
   poetry install
   ```

5. **Build Docker image for code interpreter**
   ```bash
   docker build -t mas-code-interpreter:latest -f docker/Dockerfile .
   ```

6. **Configure environment variables**
   ```bash
   cp .env.template .env
   # Edit .env and add your API keys
   ```

### Required API Keys

Edit [.env](.env) to include:
- `OPENAI_API_KEY`: OpenAI API access
- `PINECONE_API_KEY`: Pinecone for RAG experiments
- `PINECONE_ENVIRONMENT`: Pinecone environment/region
- `PINECONE_INDEX`: Index name (default: `mas-realm`)

## Running Experiments

### Basic Usage

```bash
python run_experiments_fixed.py --experiments P1 P4 --seeds 42 --model-config openai
```

### Command-Line Arguments

- `--experiments`: REALM-Bench scenario IDs (e.g., P1, P4, P7, P11)
- `--seeds`: Random seed values for reproducibility (default: 42)
- `--model-config`: Model provider configuration
  - `openai` (default): OpenAI GPT models
  - `anthropic`: Anthropic Claude models
  - `groq`: Groq inference API
- `--verbose`: Enable extended debugging output

### Example Experiments

**Baseline experiment (no RAG)**
```bash
python run_experiments_fixed.py \
  --experiments P1 P4 P7 P11 \
  --seeds 42 43 44 45 46 \
  --model-config openai
```

**RAG-augmented experiment**
```bash
# Edit configs/experiments/rag.yaml to enable RAG
python run_experiments_fixed.py \
  --experiments P1 P4 P7 P11 \
  --seeds 42 43 44 45 46 \
  --model-config openai
```

**Verbose debugging**
```bash
python run_experiments_fixed.py --experiments P1 --verbose
```

## Configuration

Configuration uses a hierarchical YAML system with environment variable overrides.

### Configuration Files

- [configs/base.yaml](configs/base.yaml): Base configuration with all parameters
- [configs/models/openai.yaml](configs/models/openai.yaml): OpenAI model settings
- [configs/models/anthropic.yaml](configs/models/anthropic.yaml): Anthropic model settings
- [configs/experiments/baseline.yaml](configs/experiments/baseline.yaml): Baseline (no RAG)
- [configs/experiments/rag.yaml](configs/experiments/rag.yaml): RAG-augmented

### Key Configuration Parameters

```yaml
model:
  provider: openai
  name: gpt-4o
  temperature: 0.7
  max_tokens: 4096

agents:
  max_workers: 10
  timeout_seconds: 300

tools:
  code_interpreter:
    enabled: true
    docker_image: mas-code-interpreter:latest

  rag:
    enabled: false  # Set to true for RAG experiments
    mode: planning_only  # or planning_and_execution
```

## Results & Analysis

### Output Structure

Experiment results are saved to `out/<experiment_name>/`:

```
out/final_experiment_run_GPT-5-mini/
├── realm_bench_experiment_summary.json    # High-level summary
├── realm_bench_detailed_results.json      # Per-task details
└── advanced_metrics_analysis.json         # Computed metrics
```

### Metrics Tracked

- **Success Metrics**: Goal achievement rate, constraint satisfaction
- **Performance Metrics**: Execution time, token usage (input/output/total)
- **Planning Metrics**: Planning latency, plan quality, parallelism, redundancy
- **Coordination Metrics**: Agent task fit, dependency management
- **Knowledge Metrics**: RAG query count, document retrieval stats

### Statistical Analysis

```bash
python stat-test.py out/<experiment_name>/advanced_metrics_analysis.json
```

Performs:
- Paired t-tests comparing RAG vs. baseline
- Effect size calculations (Cohen's d)
- Confidence intervals
- Statistical significance testing

### Metrics Analysis

```bash
python scripts/analyze_metrics.py out/<experiment_name>/
```

Generates:
- Aggregated metrics across scenarios
- Visualizations and plots
- Comparative analysis reports

## Agent Capabilities

The system supports seven specialized agent types:

| Capability | Description |
|------------|-------------|
| `GENERAL` | General-purpose problem solving and reasoning |
| `LOGISTICS` | Transportation, routing, supply chain optimization |
| `SCHEDULER` | Temporal planning, task scheduling, timeline management |
| `RESOURCE_MANAGER` | Resource allocation, capacity planning, inventory management |
| `OPTIMIZER` | Mathematical optimization, trade-off analysis |
| `VALIDATOR` | Verification, constraint checking, compliance validation |
| `DATA_ANALYST` | Data processing, statistical analysis, insights generation |

## Agent Tools

### Code Interpreter

Docker-based sandboxed Python execution environment with:
- Scientific libraries: NumPy, Pandas, Matplotlib, SciPy, scikit-learn
- Jupyter kernel for code execution
- Isolated execution for security

### RAG Tool

Knowledge retrieval using:
- Vector database: Pinecone
- Embeddings: sentence-transformers
- Query strategies: Multiple queries per task for comprehensive knowledge gathering

## Technology Stack

- **Orchestration**: LangGraph 0.2.0+, LangChain 0.3.0+
- **Models**: OpenAI via API
- **Data Validation**: Pydantic 2.5.0+
- **Code Execution**: Docker, Jupyter
- **Vector Database**: Pinecone, ChromaDB
- **Analysis**: Pandas, NumPy, Matplotlib, SciPy
- **Testing**: pytest, pytest-asyncio
- **Code Quality**: Black (100 char), Ruff, mypy (strict)

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/ --line-length 100
```

### Linting

```bash
ruff check src/ scripts/
```

### Type Checking

```bash
mypy src/ --strict
```

## Troubleshooting

### Common Issues

**Docker image not found**
```bash
# Rebuild the code interpreter image
docker build -t mas-code-interpreter:latest -f docker/Dockerfile .
```

**API key errors**
```bash
# Verify .env file has correct API keys
cat .env | grep API_KEY
```

**Import errors from REALM-Bench**
```bash
# Ensure REALM-Bench is cloned in correct location
ls third_party/realm_bench/
```

**Model-specific issues**
- OpenAI: Ensure API key has access to GPT-4o models
- Anthropic/Groq: Currently not working, use `openai` model config

## Contact

For questions or issues related to this research, please contact the authors.
