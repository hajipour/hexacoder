# HexaCoder: Secure Code Generation via Oracle-Guided Synthetic Training Data

## Installation

```bash
git clone <repository-url>
cd hexacoder
pip install -r requirements.txt
pip install -e .
```

### Prerequisites

- Python 3.7+
- CodeQL CLI (for vulnerability analysis)
- Docker/Podman (for containerized evaluation)

## Usage

### Running CodeQL Analysis
s
```bash
python run_codeql.py --checkpoint <model_checkpoint>
```

This will analyze generated code samples for security vulnerabilities across supported CWE categories.

### Model Fine-tuning

```bash
cd finetune
python finetune.py
```

Fine-tune code generation models on security-focused datasets.

### Functional Evaluation

```bash
cd eval_functional
python human_eval_exec.py --output_name <experiment_name> --max_workers 50
```

Evaluate code functionality using containerized execution.

## Project Structure

```
hexacoder/
├── benchmark_data/          # Security benchmark datasets
│   ├── codelmsec/          # CodeLMSec benchmark
│   ├── pearce/             # Pearce et al. benchmark
│   └── gen/                # Generated samples
├── eval_functional/         # Functional evaluation tools
├── finetune/               # Model training scripts
├── dataset/                # Training datasets
└── run_codeql.py          # CodeQL analysis runner
```

## Benchmark Data

The project includes comprehensive benchmark datasets:

- **CodeLMSec**: Industry-standard security benchmarks for C and Python
- **Pearce Dataset**: Academic research benchmarks
- **Generated Samples**: Model-generated code for evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use HexaCoder in your research, please cite:

```bibtex
@software{hexacoder,
  title={HexaCoder: Security-Focused Code Analysis Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/hexacoder}
}
```

## Acknowledgments

- CodeQL team for the static analysis engine
- CodeLMSec and Pearce et al. for benchmark datasets
- The security research community for CWE classifications
