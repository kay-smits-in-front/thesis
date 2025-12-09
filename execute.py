import subprocess
import sys

def run_module(module_name):
	try:
		result = subprocess.run(
			[sys.executable, "-m", module_name],
			check=True
		)
		return True
	except subprocess.CalledProcessError:
		return False

def main():
	modules = [
		"thesis_models.physics.mlp",
		"thesis_models.initial.MLR",
		"thesis_models.physics.rnn",
		"thesis_models.physics.lstm"
	]

	failed = []

	for module in modules:
		print(f"\n{'='*60}")
		print(f"Starting: {module}")
		print(f"{'='*60}\n")

		success = run_module(module)

		if not success:
			failed.append(module)

	print(f"\n{'='*60}")
	print("Summary")
	print(f"{'='*60}")
	print(f"Total modules: {len(modules)}")
	print(f"Failed: {len(failed)}")

	if failed:
		print("\nFailed modules:")
		for module in failed:
			print(f"  - {module}")

if __name__ == "__main__":
	main()