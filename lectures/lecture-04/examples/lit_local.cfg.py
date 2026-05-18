import os
# Minimal lit config template for local experiments.
import lit.formats

config.name = 'Lecture04-MLIR-NPU'
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root
config.substitutions.append(('%PATH%', os.environ.get('PATH', '')))
