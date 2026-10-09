"""Use the full-development refit with the shared inference implementation."""
import sys
from pathlib import Path
from predict_improved import main

if __name__=='__main__':
    if '--checkpoint' not in sys.argv:
        sys.argv.extend(['--checkpoint',str(Path(__file__).resolve().parent/'outputs/improvement_01/refit/best_system.pt')])
    main()
