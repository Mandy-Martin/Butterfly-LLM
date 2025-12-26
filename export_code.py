#!/usr/bin/env python3
"""
Simple script to export all code files into a single output file
Similar to repomix functionality
"""
import os
from pathlib import Path

# Files and directories to ignore
IGNORE_PATTERNS = {
    '__pycache__',
    '.git',
    '.vscode',
    'node_modules',
    'venv',
    'env',
    '.pytest_cache',
    '*.pyc',
    'checkpoint.pt',
    'optim.pt',
    'repomix-output.*',
    'export_code.py'
}

# File extensions to include
CODE_EXTENSIONS = {'.py', '.yaml', '.yml', '.txt', '.md', '.json', '.js', '.ts', '.html', '.css'}

def should_ignore(path):
    """Check if a path should be ignored"""
    path_str = str(path)
    for pattern in IGNORE_PATTERNS:
        if pattern in path_str:
            return True
    return False

def export_code(output_file='repomix-output.txt', style='text'):
    """Export all code files to a single output file"""
    project_root = Path('.')
    output_lines = []
    
    # Get all files
    code_files = []
    for root, dirs, files in os.walk('.'):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if not should_ignore(Path(root) / d)]
        
        for file in files:
            file_path = Path(root) / file
            if should_ignore(file_path):
                continue
            
            # Include code files
            if file_path.suffix in CODE_EXTENSIONS or file_path.name in ['README', 'LICENSE', '.gitignore']:
                code_files.append(file_path)
    
    # Sort files for consistent output
    code_files.sort()
    
    # Write output
    for file_path in code_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if style == 'text':
                output_lines.append(f"\n{'='*80}\n")
                output_lines.append(f"File: {file_path}\n")
                output_lines.append(f"{'='*80}\n\n")
                output_lines.append(content)
                output_lines.append("\n")
            elif style == 'markdown':
                output_lines.append(f"\n## {file_path}\n\n")
                output_lines.append("```\n")
                output_lines.append(content)
                output_lines.append("\n```\n")
            elif style == 'xml':
                # Escape XML special characters
                content_escaped = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                output_lines.append(f"\n<file path=\"{file_path}\">\n")
                output_lines.append(f"<![CDATA[\n{content}\n]]>\n")
                output_lines.append("</file>\n")
        
        except Exception as e:
            output_lines.append(f"\nError reading {file_path}: {e}\n")
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(output_lines))
    
    print(f"Code exported to {output_file}")
    print(f"  Total files: {len(code_files)}")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    output_file = 'repomix-output.txt'
    style = 'text'
    
    if '--style' in sys.argv:
        idx = sys.argv.index('--style')
        if idx + 1 < len(sys.argv):
            style = sys.argv[idx + 1]
            if style == 'markdown':
                output_file = 'repomix-output.md'
            elif style == 'xml':
                output_file = 'repomix-output.xml'
            elif style == 'json':
                output_file = 'repomix-output.json'
    
    export_code(output_file, style)

