"""
Run all three model optimizations
"""

import subprocess
import sys

scripts = [
	('LSTM', 'thesis_models/physics/lstm_optimization_clean.py'),
	('RNN', 'thesis_models/physics/rnn_optimization_clean.py'),
	('MLP', 'thesis_models/physics/mlp_optimization_clean.py')
]

for name, script in scripts:
	print(f"\n{'='*80}")
	print(f"RUNNING {name} ")
	print(f"{'='*80}\n")

	try:
		subprocess.run([sys.executable, script], check=True)
		print(f"\n✓ {name} completed successfully")
	except subprocess.CalledProcessError as e:
		print(f"\n✗ {name} failed: {e}")
		user_input = input("Continue? (y/n): ")
		if user_input.lower() != 'y':
			break
	except KeyboardInterrupt:
		print(f"\n✗ {name} interrupted by user")
		break

print("\n" + "="*80)
print("ALL runs COMPLETE")
print("="*80)