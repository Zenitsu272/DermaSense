"""Connect a generated evaluation kit to an existing, prepared Python environment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-dir', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    folder = args.demo_dir.resolve()
    python = args.python.resolve()
    notebook = folder / 'DermaSense_Live_Demo.ipynb'
    if not notebook.is_file() or not python.is_file():
        parser.error('Provide the generated evaluation folder and a valid Python executable.')

    # Fail before changing settings if the selected environment is incomplete.
    subprocess.run([str(python), '-c',
        'import torch, torchvision, numpy, PIL, sklearn, pandas, matplotlib, ipywidgets, ipykernel; '
        'print("Environment ready; CUDA available:", torch.cuda.is_available())'], check=True)
    subprocess.run([str(python), '-m', 'ipykernel', 'install', '--user',
                    '--name', 'dermasense-demo', '--display-name', 'DermaSense Demo (GPU)'], check=True)

    config = folder / '.vscode'
    config.mkdir(exist_ok=True)
    settings_path = config / 'settings.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8')) if settings_path.exists() else {}
    settings.update({'python.defaultInterpreterPath': str(python),
                     'jupyter.notebookFileRoot': '${fileDirname}'})
    settings_path.write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    data = json.loads(notebook.read_text(encoding='utf-8'))
    data['metadata']['kernelspec'] = {'display_name': 'DermaSense Demo (GPU)',
                                    'language': 'python', 'name': 'dermasense-demo'}
    notebook.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding='utf-8')
    (folder / 'Start_VSCode.cmd').write_bytes(
        b'@echo off\r\ncd /d "%~dp0"\r\n'
        b'call code --new-window "%~dp0." "%~dp0DermaSense_Live_Demo.ipynb"\r\n'
        b'if errorlevel 1 pause\r\n')
    print('Open the evaluation FOLDER in VS Code, then open the notebook.')
    print('Select Another Kernel > Jupyter Kernel > DermaSense Demo (GPU).')
    print('CPU fallback remains available. Model weights and notebook cells are unchanged.')


if __name__ == '__main__':
    main()
