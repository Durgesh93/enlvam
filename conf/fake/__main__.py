import sys
from .__init__ import experiment_name

def main() -> None:
    print(experiment_name(), file=sys.stdout)
    sys.exit(0)

if __name__ == "__main__":
    main()
